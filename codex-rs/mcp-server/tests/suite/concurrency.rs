use std::fs;
use std::path::Path;

use codex_mcp_server::CodexToolCallParam;
use mcp_test_support::McpProcess;
use mcp_test_support::create_final_assistant_message_sse_response;
use mcp_test_support::create_mock_chat_completions_server;
use mcp_types::JSONRPCResponse;
use mcp_types::RequestId;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::timeout;
use uuid::Uuid;
use wiremock::MockServer;

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Helper struct to keep mock server and temp dir alive for the test duration
struct TestContext {
    codex_home: TempDir,
    #[allow(dead_code)]
    mock_server: MockServer,
}

/// Create a config.toml that points to the mock server
fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    let config_toml = codex_home.join("config.toml");
    std::fs::write(
        config_toml,
        format!(
            r#"
model = "mock-model"
approval_policy = "untrusted"
sandbox_policy = "read-only"

model_provider = "mock_provider"

[model_providers.mock_provider]
name = "Mock provider for test"
base_url = "{server_uri}/v1"
"#
        ),
    )
}

/// Test that multiple concurrent requests to the same conversation_id work correctly
/// without causing race conditions or duplicate writers.
/// This verifies that the per-conversation locking mechanism (from commit c6ae41c) works.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_access_same_conversation_id() {
    // Create mock server with responses for all concurrent requests
    let num_requests = 4;
    let mut responses = Vec::new();
    for i in 0..num_requests {
        responses.push(
            create_final_assistant_message_sse_response(&format!("Response to concurrent request {}", i))
                .expect("create sse response")
        );
    }
    let mock_server = create_mock_chat_completions_server(responses).await;

    // Prepare a temporary CODEX_HOME with a fake rollout file
    let codex_home = TempDir::new().expect("create temp dir");
    create_config_toml(codex_home.path(), &mock_server.uri()).expect("create config");

    let conversation_uuid = Uuid::new_v4();
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T14-30-00",
        "2025-01-15T14:30:00Z",
        conversation_uuid,
        "Initial conversation",
    );

    // Start MCP process
    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    // Send 4 concurrent requests to the same conversation_id
    let conversation_id_str = conversation_uuid.to_string();

    let mut request_ids = Vec::new();
    for i in 0..num_requests {
        let req_id = mcp
            .send_codex_tool_call(CodexToolCallParam {
                prompt: format!("Concurrent request {}", i),
                conversation_id: Some(conversation_id_str.clone()),
                resume_last_session: None,
                cwd: None,
            })
            .await
            .expect("send codex tool call");
        request_ids.push(req_id);
    }

    // Wait for all responses (sequentially, but requests were sent concurrently)
    let mut success_count = 0;
    for (i, req_id) in request_ids.iter().enumerate() {
        let resp: JSONRPCResponse = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_response_message(RequestId::Integer(*req_id)),
        )
        .await
        .expect(&format!("request {} timeout", i))
        .expect(&format!("request {} failed", i));

        // Check for successful response (has content)
        if let Some(content) = resp.result.get("content") {
            if content.is_array() && !content.as_array().unwrap().is_empty() {
                success_count += 1;
            }
        }
    }

    assert_eq!(
        success_count, num_requests,
        "All {} concurrent requests should succeed",
        num_requests
    );

    // Verify rollout file integrity - it should exist and be parseable
    let rollout_path = find_rollout_file(codex_home.path(), conversation_uuid);
    assert!(
        rollout_path.exists(),
        "Rollout file should still exist after concurrent access"
    );

    // Verify file can be read and parsed (not corrupted)
    let content = fs::read_to_string(&rollout_path).expect("read rollout file");
    let lines: Vec<&str> = content.lines().collect();
    assert!(
        lines.len() >= 3,
        "Rollout file should have at least the initial entries"
    );

    // Each line should be valid JSON
    for (i, line) in lines.iter().enumerate() {
        serde_json::from_str::<serde_json::Value>(line).expect(&format!(
            "Line {} should be valid JSON (file not corrupted)",
            i
        ));
    }
}

/// Test that creating a conversation and immediately accessing it doesn't cause
/// a race condition where the conversation is not yet persisted to disk.
/// This verifies the atomic creation fix (from commit ea75606a).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_create_then_immediate_access_race() {
    // Create mock server: 1 response for create + 3 for immediate access
    let num_immediate_requests = 3;
    let mut responses = Vec::new();
    responses.push(
        create_final_assistant_message_sse_response("Created new conversation")
            .expect("create sse response")
    );
    for i in 0..num_immediate_requests {
        responses.push(
            create_final_assistant_message_sse_response(&format!("Response to immediate access {}", i))
                .expect("create sse response")
        );
    }
    let mock_server = create_mock_chat_completions_server(responses).await;

    let codex_home = TempDir::new().expect("create temp dir");
    create_config_toml(codex_home.path(), &mock_server.uri()).expect("create config");

    // Start MCP process
    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    // Create a new conversation
    let create_req_id = mcp
        .send_codex_tool_call(CodexToolCallParam {
            prompt: "Create new conversation".to_string(),
            conversation_id: None,
            resume_last_session: Some(false),
            cwd: None,
        })
        .await
        .expect("send create request");

    // Read the response to get the conversation_id
    let create_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(create_req_id)),
    )
    .await
    .expect("create timeout")
    .expect("create response");

    // Extract conversation_id from structured_content
    let conversation_id_str = create_resp
        .result
        .get("structured_content")
        .and_then(|sc| sc.get("conversation_id"))
        .and_then(|id| id.as_str())
        .expect("Should have conversation_id in structured_content");

    let conversation_uuid =
        Uuid::parse_str(conversation_id_str).expect("conversation_id should be valid UUID");

    // IMMEDIATELY (within same event loop iteration) send multiple requests to access it
    // This tests the race condition where SessionMeta might not be flushed yet
    let mut immediate_request_ids = Vec::new();

    for i in 0..num_immediate_requests {
        let req_id = mcp
            .send_codex_tool_call(CodexToolCallParam {
                prompt: format!("Immediate access {}", i),
                conversation_id: Some(conversation_id_str.to_string()),
                resume_last_session: None,
                cwd: None,
            })
            .await
            .expect("send immediate access request");
        immediate_request_ids.push(req_id);
    }

    // Wait for all immediate access responses (sequentially, but requests were sent concurrently)
    for (i, req_id) in immediate_request_ids.iter().enumerate() {
        let resp: JSONRPCResponse = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_response_message(RequestId::Integer(*req_id)),
        )
        .await
        .expect(&format!("immediate request {} timeout", i))
        .expect(&format!("immediate request {} failed", i));

        // Check it's not an error about conversation not found
        if let Some(content) = resp.result.get("content") {
            if let Some(content_array) = content.as_array() {
                if let Some(first_item) = content_array.first() {
                    if let Some(text) = first_item.get("text").and_then(|t| t.as_str()) {
                        let text_lower = text.to_lowercase();
                        assert!(
                            !text_lower.contains("conversation not found"),
                            "Immediate access request {} should NOT get ConversationNotFound error. \
                             This indicates the atomic creation fix is broken. Got: {}",
                            i,
                            text
                        );
                    }
                }
            }
        }
    }

    // Verify the rollout file exists and is valid
    let rollout_path = find_rollout_file(codex_home.path(), conversation_uuid);
    assert!(
        rollout_path.exists(),
        "Rollout file should exist after immediate access"
    );
}

/// Test that concurrent resume operations don't create duplicate writers to the same rollout file.
/// This verifies that the cache + per-conversation locking prevents multiple CodexConversation
/// instances from writing to the same file.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_resume_prevents_duplicate_writers() {
    // Create mock server with responses for all concurrent resume requests
    let num_requests = 6;
    let mut responses = Vec::new();
    for i in 0..num_requests {
        responses.push(
            create_final_assistant_message_sse_response(&format!("Response to test message {}", i))
                .expect("create sse response")
        );
    }
    let mock_server = create_mock_chat_completions_server(responses).await;

    let codex_home = TempDir::new().expect("create temp dir");
    create_config_toml(codex_home.path(), &mock_server.uri()).expect("create config");

    let conversation_uuid = Uuid::new_v4();
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T14-30-00",
        "2025-01-15T14:30:00Z",
        conversation_uuid,
        "Original conversation",
    );

    // Start MCP process
    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    let conversation_id_str = conversation_uuid.to_string();

    // Record initial rollout file state
    let rollout_path = find_rollout_file(codex_home.path(), conversation_uuid);
    let initial_content = fs::read_to_string(&rollout_path).expect("read initial rollout");
    let initial_line_count = initial_content.lines().count();

    // Send many concurrent requests with small delays between them
    // This increases the chance of catching a race condition where multiple
    // CodexConversation instances would be created for the same conversation_id
    let mut request_ids = Vec::new();

    for i in 0..num_requests {
        let req_id = mcp
            .send_codex_tool_call(CodexToolCallParam {
                prompt: format!("Test message {}", i),
                conversation_id: Some(conversation_id_str.clone()),
                resume_last_session: None,
                cwd: None,
            })
            .await
            .expect("send concurrent resume request");
        request_ids.push(req_id);

        // Small delay to stagger requests slightly (makes race condition more likely if not fixed)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Wait for all responses (sequentially, but requests were sent concurrently)
    let mut success_count = 0;
    for (i, req_id) in request_ids.iter().enumerate() {
        let resp: JSONRPCResponse = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_response_message(RequestId::Integer(*req_id)),
        )
        .await
        .expect(&format!("concurrent resume {} timeout", i))
        .expect(&format!("concurrent resume {} failed", i));

        if let Some(content) = resp.result.get("content") {
            if content.is_array() && !content.as_array().unwrap().is_empty() {
                success_count += 1;
            }
        }
    }

    assert_eq!(
        success_count, num_requests,
        "All concurrent resume requests should succeed"
    );

    // CRITICAL CHECK: Verify rollout file integrity
    let final_content = fs::read_to_string(&rollout_path).expect("read final rollout");
    let final_lines: Vec<&str> = final_content.lines().collect();

    // File should have grown (new entries added)
    assert!(
        final_lines.len() > initial_line_count,
        "Rollout file should have new entries"
    );

    // CRITICAL: Every line must be valid JSON (if there were duplicate writers
    // and they interleaved writes, we'd get corrupted JSON)
    for (i, line) in final_lines.iter().enumerate() {
        serde_json::from_str::<serde_json::Value>(line).unwrap_or_else(|e| {
            panic!(
                "Line {} is corrupted JSON, indicating duplicate writers! Error: {}\nLine: {}",
                i, e, line
            )
        });
    }

    // Check that the file doesn't have weird interleaving patterns
    // (e.g., partial JSON objects that would indicate concurrent unbuffered writes)
    let all_lines_start_with_brace = final_lines.iter().all(|line| line.trim().starts_with('{'));
    assert!(
        all_lines_start_with_brace,
        "All lines should start with '{{' (valid JSON objects). \
         If not, duplicate writers may have interleaved writes."
    );
}

// Helper function to create a fake rollout file with a specific UUID
fn create_fake_rollout_with_uuid(
    codex_home: &Path,
    filename_ts: &str,
    meta_rfc3339: &str,
    uuid: Uuid,
    preview: &str,
) {
    // sessions/YYYY/MM/DD/ derived from filename_ts (YYYY-MM-DDThh-mm-ss)
    let year = &filename_ts[0..4];
    let month = &filename_ts[5..7];
    let day = &filename_ts[8..10];
    let dir = codex_home.join("sessions").join(year).join(month).join(day);
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create sessions dir: {e}"));

    let file_path = dir.join(format!("rollout-{filename_ts}-{uuid}.jsonl"));
    let mut lines = Vec::new();

    // Meta line with timestamp (flattened meta in payload for new schema)
    lines.push(
        json!({
            "timestamp": meta_rfc3339,
            "type": "session_meta",
            "payload": {
                "id": uuid,
                "timestamp": meta_rfc3339,
                "cwd": "/",
                "originator": "codex",
                "cli_version": "0.0.0",
                "instructions": null
            }
        })
        .to_string(),
    );

    // Minimal user message entry as a persisted response item (with envelope timestamp)
    lines.push(
        json!({
            "timestamp": meta_rfc3339,
            "type":"response_item",
            "payload": {
                "type":"message",
                "role":"user",
                "content":[{"type":"input_text","text": preview}]
            }
        })
        .to_string(),
    );

    // Add a matching user message event line to satisfy filters
    lines.push(
        json!({
            "timestamp": meta_rfc3339,
            "type":"event_msg",
            "payload": {
                "type":"user_message",
                "message": preview,
                "kind": "plain"
            }
        })
        .to_string(),
    );

    fs::write(file_path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("write rollout file: {e}"));
}

// Helper function to find the rollout file for a given conversation UUID
fn find_rollout_file(codex_home: &Path, uuid: Uuid) -> std::path::PathBuf {
    let sessions_dir = codex_home.join("sessions");

    // Walk through YYYY/MM/DD structure
    for year_entry in fs::read_dir(&sessions_dir).expect("read sessions dir") {
        let year_path = year_entry.expect("year entry").path();
        if !year_path.is_dir() {
            continue;
        }

        for month_entry in fs::read_dir(&year_path).expect("read year dir") {
            let month_path = month_entry.expect("month entry").path();
            if !month_path.is_dir() {
                continue;
            }

            for day_entry in fs::read_dir(&month_path).expect("read month dir") {
                let day_path = day_entry.expect("day entry").path();
                if !day_path.is_dir() {
                    continue;
                }

                for file_entry in fs::read_dir(&day_path).expect("read day dir") {
                    let file_path = file_entry.expect("file entry").path();
                    if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                        if filename.contains(&uuid.to_string()) {
                            return file_path;
                        }
                    }
                }
            }
        }
    }

    panic!("Could not find rollout file for UUID: {}", uuid);
}
