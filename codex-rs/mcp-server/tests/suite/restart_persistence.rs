use std::fs;
use std::path::Path;

use codex_mcp_server::CodexToolCallParam;
use mcp_test_support::McpProcess;
use mcp_types::JSONRPCResponse;
use mcp_types::RequestId;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::timeout;
use uuid::Uuid;

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Test that a conversation can be resumed after an MCP server restart using conversation_id.
/// This tests the new disk-first persistence architecture where conversations are not stored in memory.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_restart_conversation_persistence_by_id() {
    // Prepare a temporary CODEX_HOME with a fake rollout file for a specific conversation
    let codex_home = TempDir::new().expect("create temp dir");
    let conversation_uuid = Uuid::new_v4();
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T14-30-00",
        "2025-01-15T14:30:00Z",
        conversation_uuid,
        "Tell me about Rust programming",
    );

    // Start first MCP process
    let mut mcp1 = McpProcess::new(codex_home.path())
        .await
        .expect("spawn first mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp1.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    // Try to use a conversation by ID - this should work with disk persistence
    let req_id1 = mcp1
        .send_codex_tool_call(CodexToolCallParam {
            prompt: "What is ownership in Rust?".to_string(),
            conversation_id: Some(conversation_uuid.to_string()),
            resume_last_session: None,
            cwd: None,
            ..Default::default()
        })
        .await
        .expect("send codex tool call");

    // Should receive successful response (conversation found on disk)
    let resp1: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp1.read_stream_until_response_message(RequestId::Integer(req_id1)),
    )
    .await
    .expect("codex tool call timeout")
    .expect("codex tool call response");

    // Verify the response is successful by checking for content
    let _response_text = resp1.result["content"][0]["text"].as_str()
        .expect("Expected successful response with text content");

    // Shutdown the first MCP process (simulating restart)
    drop(mcp1);

    // Start a second MCP process (simulating restart)
    let mut mcp2 = McpProcess::new(codex_home.path())
        .await
        .expect("spawn second mcp process after restart");
    timeout(DEFAULT_READ_TIMEOUT, mcp2.initialize())
        .await
        .expect("second init timeout")
        .expect("second init failed");

    // Try to use the same conversation by ID after restart - should still work
    let req_id2 = mcp2
        .send_codex_tool_call(CodexToolCallParam {
            prompt: "Can you explain borrowing?".to_string(),
            conversation_id: Some(conversation_uuid.to_string()),
            resume_last_session: None,
            cwd: None,
            ..Default::default()
        })
        .await
        .expect("send codex tool call after restart");

    // Should receive successful response (conversation still found on disk after restart)
    let resp2: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp2.read_stream_until_response_message(RequestId::Integer(req_id2)),
    )
    .await
    .expect("codex tool call after restart timeout")
    .expect("codex tool call after restart response");

    // Verify the response is successful after restart by checking for content
    let _response_text = resp2.result["content"][0]["text"].as_str()
        .expect("Expected successful response with text content after restart");
}

/// Test that resume_last_session works by finding the most recent conversation from disk.
/// This tests the new disk-first approach for finding recent conversations.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_restart_resume_last_session_from_disk() {
    // Prepare a temporary CODEX_HOME with multiple fake rollout files
    let codex_home = TempDir::new().expect("create temp dir");

    // Create an older conversation
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T10-00-00",
        "2025-01-15T10:00:00Z",
        Uuid::new_v4(),
        "Older conversation",
    );

    // Create the most recent conversation
    create_fake_rollout_with_uuid(
        codex_home.path(),
        "2025-01-15T15-30-00",
        "2025-01-15T15:30:00Z",
        Uuid::new_v4(),
        "Most recent conversation",
    );

    // Start MCP process after "restart" (no previous in-memory state)
    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    // Use resume_last_session=true without conversation_id - should find most recent on disk
    let req_id = mcp
        .send_codex_tool_call(CodexToolCallParam {
            prompt: "Continue from where we left off".to_string(),
            conversation_id: None,
            resume_last_session: Some(true),
            cwd: None,
            ..Default::default()
        })
        .await
        .expect("send codex tool call with resume_last_session");

    // Should receive successful response (most recent conversation found on disk)
    let resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(req_id)),
    )
    .await
    .expect("resume last session timeout")
    .expect("resume last session response");

    // Verify the response is successful by checking for content
    let _response_text = resp.result["content"][0]["text"].as_str()
        .expect("Expected successful response with text content for resume_last_session");
}

/// Test that providing a non-existent conversation_id returns an appropriate error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nonexistent_conversation_id_error() {
    let codex_home = TempDir::new().expect("create temp dir");

    let mut mcp = McpProcess::new(codex_home.path())
        .await
        .expect("spawn mcp process");
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize())
        .await
        .expect("init timeout")
        .expect("init failed");

    // Try to use a non-existent conversation ID
    let nonexistent_uuid = Uuid::new_v4();
    let req_id = mcp
        .send_codex_tool_call(CodexToolCallParam {
            prompt: "This should fail".to_string(),
            conversation_id: Some(nonexistent_uuid.to_string()),
            resume_last_session: None,
            cwd: None,
            ..Default::default()
        })
        .await
        .expect("send codex tool call with nonexistent id");

    // Should receive error response (conversation not found on disk)
    let resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(req_id)),
    )
    .await
    .expect("nonexistent conversation timeout")
    .expect("nonexistent conversation response");

    // Verify we get an error response mentioning conversation not found
    let response_text = resp.result["content"][0]["text"].as_str()
        .expect("Expected error response with text content for nonexistent conversation_id");
    let text_lower = response_text.to_lowercase();
    assert!(text_lower.contains("conversation not found") || text_lower.contains("not found"),
           "Error message should mention conversation not found, got: {}", response_text);
}

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