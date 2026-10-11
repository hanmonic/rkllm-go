package api

import "net/http"

// handleEmbeddings: the RKLLM runtime has no embedding API; v2 drops v1's
// hidden-state experiment and answers 501.
func (h *Handler) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "embeddings are not supported by the RKLLM runtime", "invalid_request_error", "embeddings_not_supported")
}
