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

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

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
wire_api = "chat"
request_max_retries = 0
stream_max_retries = 0
"#
        ),
    )
}

/// Test that multiple sequential requests to the same conversation_id work correctly
/// and don't corrupt the rollout file. CodexConversation serializes turns internally,
/// so requests execute one after another, not concurrently.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sequential_requests_to_same_conversation() {
    let num_requests = 3;
    let mut responses = Vec::new();
    for i in 0..num_requests {
        responses.push(
            create_final_assistant_message_sse_response(&format!("Response {}", i))
                .expect("create sse response")
        );
    }
    let _mock_server = create_mock_chat_completions_server(responses).await;

    let codex_home = TempDir::new().expect("create temp dir");
    create_config_toml(codex_home.path(), &_mock_server.uri()).expect("create config");

    let conversation_uuid = Uuid::new_v4();
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T14-30-00",
        "2025-01-15T14:30:00Z",
        conversation_uuid,
        "Initial conversation",
    );

    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    let conversation_id_str = conversation_uuid.to_string();

    // Send requests sequentially and verify each one succeeds
    for i in 0..num_requests {
        let req_id = mcp
            .send_codex_tool_call(CodexToolCallParam {
                prompt: format!("Request {}", i),
                conversation_id: Some(conversation_id_str.clone()),
                resume_last_session: None,
                cwd: None,
            })
            .await
            .expect("send codex tool call");

        let resp: JSONRPCResponse = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_response_message(RequestId::Integer(req_id)),
        )
        .await
        .unwrap_or_else(|_| panic!("request {} timeout", i))
        .unwrap_or_else(|e| panic!("request {} failed: {}", i, e));

        // Verify successful response
        assert!(resp.result.get("content").and_then(|c| c.as_array()).map(|a| !a.is_empty()).unwrap_or(false),
                "Response {} should have non-empty content", i);
    }

    // Verify rollout file integrity - should not be corrupted by sequential writes
    let rollout_path = find_rollout_file(codex_home.path(), conversation_uuid);
    assert!(rollout_path.exists(), "Rollout file should exist");

    let content = fs::read_to_string(&rollout_path).expect("read rollout file");
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 3, "Rollout file should have entries from multiple requests");

    // Every line must be valid JSON (proof that writes weren't corrupted/interleaved)
    for (i, line) in lines.iter().enumerate() {
        serde_json::from_str::<serde_json::Value>(line).expect(&format!(
            "Line {} should be valid JSON (file not corrupted)", i
        ));
    }
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

    // Meta line
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

    // User message entry
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

    // User message event
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

fn find_rollout_file(codex_home: &Path, uuid: Uuid) -> std::path::PathBuf {
    let sessions_dir = codex_home.join("sessions");
    for year_entry in fs::read_dir(&sessions_dir).unwrap() {
        let year_path = year_entry.unwrap().path();
        if !year_path.is_dir() {
            continue;
        }
        for month_entry in fs::read_dir(&year_path).unwrap() {
            let month_path = month_entry.unwrap().path();
            if !month_path.is_dir() {
                continue;
            }
            for day_entry in fs::read_dir(&month_path).unwrap() {
                let day_path = day_entry.unwrap().path();
                if !day_path.is_dir() {
                    continue;
                }
                for file_entry in fs::read_dir(&day_path).unwrap() {
                    let file_path = file_entry.unwrap().path();
                    if file_path.to_string_lossy().contains(&uuid.to_string()) {
                        return file_path;
                    }
                }
            }
        }
    }
    panic!("Rollout file not found for UUID {}", uuid);
}
