//! Configuration object accepted by the `codex` MCP tool-call.

use mcp_types::Tool;
use mcp_types::ToolInputSchema;
use schemars::JsonSchema;
use schemars::r#gen::SchemaSettings;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;

/// Client-supplied configuration for a `codex` tool-call.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub struct CodexToolCallParam {
    /// The *initial user prompt* to start the Codex conversation.
    pub prompt: String,

    /// Working directory for the session. If relative, it is resolved against
    /// the server process's current working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}


/// Builds a `Tool` definition (JSON schema etc.) for the Codex tool-call.
pub(crate) fn create_tool_for_codex_tool_call_param() -> Tool {
    let schema = SchemaSettings::draft2019_09()
        .with(|s| {
            s.inline_subschemas = true;
            s.option_add_null_type = false;
        })
        .into_generator()
        .into_root_schema_for::<CodexToolCallParam>();

    #[expect(clippy::expect_used)]
    let schema_value =
        serde_json::to_value(&schema).expect("Codex tool schema should serialise to JSON");

    let tool_input_schema =
        serde_json::from_value::<ToolInputSchema>(schema_value).unwrap_or_else(|e| {
            panic!("failed to create Tool from schema: {e}");
        });

    Tool {
        name: "codex".to_string(),
        title: Some("Codex".to_string()),
        input_schema: tool_input_schema,
        // TODO(mbolin): This should be defined.
        output_schema: None,
        description: Some(
            "Run a Codex session. Accepts configuration parameters matching the Codex Config struct.".to_string(),
        ),
        annotations: None,
    }
}

impl CodexToolCallParam {
    /// Returns the initial user prompt and optional working directory.
    /// The Config is now entirely managed by the server via environment variables.
    pub fn into_prompt_and_cwd(self) -> (String, Option<PathBuf>) {
        let Self { prompt, cwd } = self;
        (prompt, cwd.map(PathBuf::from))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CodexToolCallReplyParam {
    /// The conversation id for this Codex session.
    pub conversation_id: String,

    /// The *next user prompt* to continue the Codex conversation.
    pub prompt: String,
}

/// Builds a `Tool` definition for the `codex-reply` tool-call.
pub(crate) fn create_tool_for_codex_tool_call_reply_param() -> Tool {
    let schema = SchemaSettings::draft2019_09()
        .with(|s| {
            s.inline_subschemas = true;
            s.option_add_null_type = false;
        })
        .into_generator()
        .into_root_schema_for::<CodexToolCallReplyParam>();

    #[expect(clippy::expect_used)]
    let schema_value =
        serde_json::to_value(&schema).expect("Codex reply tool schema should serialise to JSON");

    let tool_input_schema =
        serde_json::from_value::<ToolInputSchema>(schema_value).unwrap_or_else(|e| {
            panic!("failed to create Tool from schema: {e}");
        });

    Tool {
        name: "codex-reply".to_string(),
        title: Some("Codex Reply".to_string()),
        input_schema: tool_input_schema,
        output_schema: None,
        description: Some(
            "Continue a Codex conversation by providing the conversation id and prompt."
                .to_string(),
        ),
        annotations: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    /// We include a test to verify the exact JSON schema as "executable
    /// documentation" for the schema. When can track changes to this test as a
    /// way to audit changes to the generated schema.
    ///
    /// Seeing the fully expanded schema makes it easier to casually verify that
    /// the generated JSON for enum types such as "approval-policy" is compact.
    /// Ideally, modelcontextprotocol/inspector would provide a simpler UI for
    /// enum fields versus open string fields to take advantage of this.
    ///
    /// As of 2025-05-04, there is an open PR for this:
    /// https://github.com/modelcontextprotocol/inspector/pull/196
    #[test]
    fn verify_codex_tool_json_schema() {
        let tool = create_tool_for_codex_tool_call_param();
        let tool_json = serde_json::to_value(&tool).expect("tool serializes");
        let expected_tool_json = serde_json::json!({
          "name": "codex",
          "title": "Codex",
          "description": "Run a Codex session. Accepts configuration parameters matching the Codex Config struct.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "cwd": {
                "description": "Working directory for the session. If relative, it is resolved against the server process's current working directory.",
                "type": "string"
              },
              "prompt": {
                "description": "The *initial user prompt* to start the Codex conversation.",
                "type": "string"
              }
            },
            "required": [
              "prompt"
            ]
          }
        });
        assert_eq!(expected_tool_json, tool_json);
    }

    #[test]
    fn verify_codex_tool_reply_json_schema() {
        let tool = create_tool_for_codex_tool_call_reply_param();
        let tool_json = serde_json::to_value(&tool).expect("tool serializes");
        let expected_tool_json = serde_json::json!({
          "description": "Continue a Codex conversation by providing the conversation id and prompt.",
          "inputSchema": {
            "properties": {
              "conversationId": {
                "description": "The conversation id for this Codex session.",
                "type": "string"
              },
              "prompt": {
                "description": "The *next user prompt* to continue the Codex conversation.",
                "type": "string"
              },
            },
            "required": [
              "conversationId",
              "prompt",
            ],
            "type": "object",
          },
          "name": "codex-reply",
          "title": "Codex Reply",
        });
        assert_eq!(expected_tool_json, tool_json);
    }
}

impl Default for CodexToolCallParam {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            cwd: None,
        }
    }
}
