use crate::AuthManager;
use crate::CodexAuth;
use crate::codex::Codex;
use crate::codex::CodexSpawnOk;
use crate::codex::INITIAL_SUBMIT_ID;
use crate::codex::compact::content_items_to_text;
use crate::codex::compact::is_session_prefix_message;
use crate::codex_conversation::CodexConversation;
use crate::config::Config;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::protocol::Event;
use crate::protocol::EventMsg;
use crate::protocol::SessionConfiguredEvent;
use crate::rollout::RolloutRecorder;
use codex_protocol::ConversationId;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::InitialHistory;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::SessionSource;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::RwLock;

/// Represents a newly created Codex conversation, including the first event
/// (which is [`EventMsg::SessionConfigured`]).
pub struct NewConversation {
    pub conversation_id: ConversationId,
    pub conversation: Arc<CodexConversation>,
    pub session_configured: SessionConfiguredEvent,
}

/// Extract the conversation ID from a rollout file path.
/// Expected filename format: `rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl`
fn extract_conversation_id_from_path(path: &Path) -> CodexResult<ConversationId> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            CodexErr::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid rollout path: {path:?}"),
            ))
        })?;

    // Expected: rollout-YYYY-MM-DDThh-mm-ss-<uuid>.jsonl
    let core = filename
        .strip_prefix("rollout-")
        .and_then(|s| s.strip_suffix(".jsonl"))
        .ok_or_else(|| {
            CodexErr::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid rollout filename format: {filename}"),
            ))
        })?;

    // Scan from the right for a '-' such that the suffix parses as a valid UUID.
    // This mirrors the logic in parse_timestamp_uuid_from_filename in rollout/list.rs
    // For example: "2025-01-15T14-30-00-31a0637d-8a72-49fd-b5ca-f7a1e331f6f6"
    // We try parsing from right: "f6f6" (fail), "f7a1e331f6f6" (fail), ...
    // until we find "31a0637d-8a72-49fd-b5ca-f7a1e331f6f6" (success)
    // Parse only once and return the ConversationId directly
    core.match_indices('-')
        .rev()
        .find_map(|(i, _)| {
            let candidate = &core[i + 1..];
            ConversationId::from_string(candidate).ok()
        })
        .ok_or_else(|| {
            CodexErr::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Cannot extract UUID from filename: {filename}"),
            ))
        })
}

/// [`ConversationManager`] is responsible for creating conversations and
/// managing them through persistent rollout files on disk.
/// Maintains an in-memory cache of active conversations to avoid repeated disk I/O
/// and prevent multiple writers to the same rollout file.
/// Uses per-conversation locks to prevent TOCTOU race conditions during resume.
pub struct ConversationManager {
    auth_manager: Arc<AuthManager>,
    session_source: SessionSource,
    cache: Arc<RwLock<HashMap<ConversationId, Arc<CodexConversation>>>>,
    /// Per-conversation locks to ensure only one resume operation runs at a time
    /// for each conversation ID. Prevents multiple writers to the same rollout file.
    resume_locks: Arc<Mutex<HashMap<ConversationId, Arc<Mutex<()>>>>>,
}

impl ConversationManager {
    pub fn new(auth_manager: Arc<AuthManager>, session_source: SessionSource) -> Self {
        Self {
            auth_manager,
            session_source,
            cache: Arc::new(RwLock::new(HashMap::new())),
            resume_locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Construct with a dummy AuthManager containing the provided CodexAuth.
    /// Used for integration tests: should not be used by ordinary business logic.
    pub fn with_auth(auth: CodexAuth) -> Self {
        Self::new(
            crate::AuthManager::from_auth_for_testing(auth),
            SessionSource::Exec,
        )
    }

    pub async fn new_conversation(&self, config: Config) -> CodexResult<NewConversation> {
        self.spawn_conversation(config, self.auth_manager.clone())
            .await
    }

    async fn spawn_conversation(
        &self,
        config: Config,
        auth_manager: Arc<AuthManager>,
    ) -> CodexResult<NewConversation> {
        let CodexSpawnOk {
            codex,
            conversation_id,
        } = Codex::spawn(
            config,
            auth_manager,
            InitialHistory::New,
            self.session_source,
        )
        .await?;
        self.finalize_spawn(codex, conversation_id).await
    }

    async fn finalize_spawn(
        &self,
        codex: Codex,
        conversation_id: ConversationId,
    ) -> CodexResult<NewConversation> {
        // The first event must be `SessionInitialized`. Validate and forward it
        // to the caller so that they can display it in the conversation
        // history.
        let event = codex.next_event().await?;
        let session_configured = match event {
            Event {
                id,
                msg: EventMsg::SessionConfigured(session_configured),
            } if id == INITIAL_SUBMIT_ID => session_configured,
            _ => {
                return Err(CodexErr::SessionConfiguredNotFirstEvent);
            }
        };

        let conversation = Arc::new(CodexConversation::new(codex));

        // Add to cache to ensure single writer per conversation
        self.cache
            .write()
            .await
            .insert(conversation_id, conversation.clone());

        Ok(NewConversation {
            conversation_id,
            conversation,
            session_configured,
        })
    }

    /// Shared helper that resumes a conversation with proper locking.
    /// If `rollout_path` is None, searches for the file by conversation ID.
    /// If `rollout_path` is Some, uses that path directly (more efficient).
    async fn resume_conversation_with_lock(
        &self,
        conversation_id: ConversationId,
        config: Config,
        rollout_path: Option<PathBuf>,
    ) -> CodexResult<Arc<CodexConversation>> {
        // Fast path: check cache first
        {
            let cache = self.cache.read().await;
            if let Some(conversation) = cache.get(&conversation_id) {
                return Ok(conversation.clone());
            }
        }

        // Slow path: acquire per-conversation lock to prevent multiple resumes
        // Get or create the lock for this conversation_id
        let conversation_lock = {
            let mut locks = self.resume_locks.lock().await;
            locks
                .entry(conversation_id)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        // Acquire the per-conversation lock - only one resume per conversation at a time
        let _guard = conversation_lock.lock().await;

        // Double-check: another task might have loaded it while we waited for the lock
        {
            let cache = self.cache.read().await;
            if let Some(conversation) = cache.get(&conversation_id) {
                return Ok(conversation.clone());
            }
        }

        // Determine the rollout path
        let rollout_path = match rollout_path {
            Some(path) => path,
            None => {
                // Search for rollout file by conversation ID
                let codex_home = &config.codex_home;
                let id_str = conversation_id.to_string();
                crate::rollout::find_conversation_path_by_id_str(codex_home, &id_str)
                    .await
                    .map_err(CodexErr::Io)?
                    .ok_or_else(|| CodexErr::ConversationNotFound(conversation_id))?
            }
        };

        let path_for_logging = rollout_path.clone();

        // Resume conversation from rollout file
        let resumed = self
            .resume_conversation_from_rollout(config, rollout_path, self.auth_manager.clone())
            .await?;

        // Verify the conversation_id matches
        if resumed.conversation_id == conversation_id {
            // finalize_spawn already added to cache, just return the conversation
            Ok(resumed.conversation)
        } else {
            tracing::error!(
                "Conversation ID mismatch: expected {}, got {} from file {:?}",
                conversation_id,
                resumed.conversation_id,
                path_for_logging
            );
            Err(CodexErr::ConversationNotFound(conversation_id))
        }
    }

    /// Get or resume a conversation from disk by its ID. This method first checks
    /// the in-memory cache, then falls back to loading from disk if needed.
    /// Uses per-conversation locking to prevent concurrent resumes of the same conversation.
    pub async fn get_or_resume_conversation(
        &self,
        conversation_id: ConversationId,
        config: Config,
    ) -> CodexResult<Arc<CodexConversation>> {
        self.resume_conversation_with_lock(conversation_id, config, None)
            .await
    }

    /// Get the most recent conversation from disk, if any exists.
    /// Extracts the conversation ID from the path and uses per-conversation locking
    /// to prevent race conditions with get_or_resume_conversation.
    pub async fn get_most_recent_conversation(
        &self,
        config: Config,
    ) -> CodexResult<Option<Arc<CodexConversation>>> {
        let codex_home = &config.codex_home;

        // Find the most recent rollout file
        let rollout_path = crate::rollout::find_most_recent_conversation_path(codex_home)
            .await
            .map_err(CodexErr::Io)?;

        match rollout_path {
            Some(path) => {
                // Extract conversation ID from the filename
                let conversation_id = extract_conversation_id_from_path(&path)?;

                // Use the shared helper with per-conversation locking
                // This prevents race conditions with get_or_resume_conversation
                let conversation = self
                    .resume_conversation_with_lock(conversation_id, config, Some(path))
                    .await?;

                Ok(Some(conversation))
            }
            None => {
                // No conversations found
                Ok(None)
            }
        }
    }

    pub async fn resume_conversation_from_rollout(
        &self,
        config: Config,
        rollout_path: PathBuf,
        auth_manager: Arc<AuthManager>,
    ) -> CodexResult<NewConversation> {
        let initial_history = RolloutRecorder::get_rollout_history(&rollout_path).await?;
        let CodexSpawnOk {
            codex,
            conversation_id,
        } = Codex::spawn(config, auth_manager, initial_history, self.session_source).await?;
        self.finalize_spawn(codex, conversation_id).await
    }

    /// Removes the conversation from the in-memory cache. The conversation is stored
    /// as `Arc<CodexConversation>`, so other references may exist elsewhere.
    /// Returns the conversation if it was found and removed from the cache.
    /// Note: This does not delete the rollout file on disk.
    pub async fn remove_conversation(
        &self,
        conversation_id: &ConversationId,
    ) -> Option<Arc<CodexConversation>> {
        self.cache.write().await.remove(conversation_id)
    }

    /// Fork an existing conversation by taking messages up to the given position
    /// (not including the message at the given position) and starting a new
    /// conversation with identical configuration (unless overridden by the
    /// caller's `config`). The new conversation will have a fresh id.
    pub async fn fork_conversation(
        &self,
        nth_user_message: usize,
        config: Config,
        path: PathBuf,
    ) -> CodexResult<NewConversation> {
        // Compute the prefix up to the cut point.
        let history = RolloutRecorder::get_rollout_history(&path).await?;
        let history = truncate_before_nth_user_message(history, nth_user_message);

        // Spawn a new conversation with the computed initial history.
        let auth_manager = self.auth_manager.clone();
        let CodexSpawnOk {
            codex,
            conversation_id,
        } = Codex::spawn(config, auth_manager, history, self.session_source).await?;

        self.finalize_spawn(codex, conversation_id).await
    }
}

/// Return a prefix of `items` obtained by cutting strictly before the nth user message
/// (0-based) and all items that follow it.
fn truncate_before_nth_user_message(history: InitialHistory, n: usize) -> InitialHistory {
    // Work directly on rollout items, and cut the vector at the nth user message input.
    let items: Vec<RolloutItem> = history.get_rollout_items();

    // Find indices of user message inputs in rollout order.
    let mut user_positions: Vec<usize> = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        if let RolloutItem::ResponseItem(ResponseItem::Message { role, content, .. }) = item
            && role == "user"
            && content_items_to_text(content).is_some_and(|text| !is_session_prefix_message(&text))
        {
            user_positions.push(idx);
        }
    }

    // If fewer than or equal to n user messages exist, treat as empty (out of range).
    if user_positions.len() <= n {
        return InitialHistory::New;
    }

    // Cut strictly before the nth user message (do not keep the nth itself).
    let cut_idx = user_positions[n];
    let rolled: Vec<RolloutItem> = items.into_iter().take(cut_idx).collect();

    if rolled.is_empty() {
        InitialHistory::New
    } else {
        InitialHistory::Forked(rolled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex::make_session_and_context;
    use assert_matches::assert_matches;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ReasoningItemReasoningSummary;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;

    fn user_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
        }
    }
    fn assistant_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: text.to_string(),
            }],
        }
    }

    #[test]
    fn drops_from_last_user_only() {
        let items = [
            user_msg("u1"),
            assistant_msg("a1"),
            assistant_msg("a2"),
            user_msg("u2"),
            assistant_msg("a3"),
            ResponseItem::Reasoning {
                id: "r1".to_string(),
                summary: vec![ReasoningItemReasoningSummary::SummaryText {
                    text: "s".to_string(),
                }],
                content: None,
                encrypted_content: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "tool".to_string(),
                arguments: "{}".to_string(),
                call_id: "c1".to_string(),
            },
            assistant_msg("a4"),
        ];

        // Wrap as InitialHistory::Forked with response items only.
        let initial: Vec<RolloutItem> = items
            .iter()
            .cloned()
            .map(RolloutItem::ResponseItem)
            .collect();
        let truncated = truncate_before_nth_user_message(InitialHistory::Forked(initial), 1);
        let got_items = truncated.get_rollout_items();
        let expected_items = vec![
            RolloutItem::ResponseItem(items[0].clone()),
            RolloutItem::ResponseItem(items[1].clone()),
            RolloutItem::ResponseItem(items[2].clone()),
        ];
        assert_eq!(
            serde_json::to_value(&got_items).unwrap(),
            serde_json::to_value(&expected_items).unwrap()
        );

        let initial2: Vec<RolloutItem> = items
            .iter()
            .cloned()
            .map(RolloutItem::ResponseItem)
            .collect();
        let truncated2 = truncate_before_nth_user_message(InitialHistory::Forked(initial2), 2);
        assert_matches!(truncated2, InitialHistory::New);
    }

    #[test]
    fn ignores_session_prefix_messages_when_truncating() {
        let (session, turn_context) = make_session_and_context();
        let mut items = session.build_initial_context(&turn_context);
        items.push(user_msg("feature request"));
        items.push(assistant_msg("ack"));
        items.push(user_msg("second question"));
        items.push(assistant_msg("answer"));

        let rollout_items: Vec<RolloutItem> = items
            .iter()
            .cloned()
            .map(RolloutItem::ResponseItem)
            .collect();

        let truncated = truncate_before_nth_user_message(InitialHistory::Forked(rollout_items), 1);
        let got_items = truncated.get_rollout_items();

        let expected: Vec<RolloutItem> = vec![
            RolloutItem::ResponseItem(items[0].clone()),
            RolloutItem::ResponseItem(items[1].clone()),
            RolloutItem::ResponseItem(items[2].clone()),
        ];

        assert_eq!(
            serde_json::to_value(&got_items).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
}
