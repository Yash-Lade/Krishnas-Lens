import { useEffect, useState } from "react";
import {
  Box,
  Typography,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from "@mui/material";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import { getMyFeedbackApi, deleteFeedbackApi } from "../../services/feedbackApi";
import { timeAgo } from "../../services/timeAgo";
import RatingStars from "./RatingStars";

export default function FeedbackList({ limit = 5 }) {
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState([]);

  // delete dialog state
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);

  const load = async () => {
    try {
      setLoading(true);
      const res = await getMyFeedbackApi();
      const data = res?.data?.data || [];
      setItems(data.slice(0, limit));
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [limit]);

  const askDelete = (feedback) => {
    setSelected(feedback);
    setOpen(true);
  };

  const confirmDelete = async () => {
    try {
      await deleteFeedbackApi(selected?._id);
      setOpen(false);
      setSelected(null);
      load();
    } catch {
      // ignore UI crash
      setOpen(false);
    }
  };

  return (
    <Box
      sx={{
        borderRadius: 4,
        p: { xs: 2, md: 2.5 },
        background: "rgba(255,255,255,0.72)",
        border: "1px solid rgba(15,23,42,0.06)",
        backdropFilter: "blur(12px)",
        boxShadow: "0 14px 32px rgba(15,23,42,0.07)",
      }}
    >
      <Typography sx={{ fontWeight: 1000, fontSize: 18 }}>
        Recent Feedback
      </Typography>

      <Typography sx={{ color: "text.secondary", fontSize: 13, mt: 0.4 }}>
        Your latest feedback linked with thoughts.
      </Typography>

      <Divider sx={{ my: 2, opacity: 0.6 }} />

      {loading ? (
        <Typography sx={{ color: "text.secondary" }}>Loading...</Typography>
      ) : !items.length ? (
        <Typography sx={{ color: "text.secondary" }}>
          No feedback submitted yet.
        </Typography>
      ) : (
        <Box sx={{ display: "grid", gap: 1.2 }}>
          {items.map((f) => (
            <Box
              key={f._id}
              sx={{
                p: 1.4,
                borderRadius: 3,
                border: "1px solid rgba(15,23,42,0.06)",
                background: "rgba(255,255,255,0.75)",
                display: "grid",
                gap: 1,
              }}
            >
              {/* top row */}
              <Box
                sx={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: 1,
                }}
              >
                <Box sx={{ minWidth: 0 }}>
                  <Typography sx={{ fontWeight: 900 }} noWrap>
                    {f?.entryId?.thoughtText || "Entry"}
                  </Typography>

                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", fontWeight: 700 }}
                  >
                    Submitted {timeAgo(f.createdAt)}
                  </Typography>
                </Box>

                <Tooltip title="Delete feedback">
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => askDelete(f)}
                  >
                    <DeleteOutlineRoundedIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>

              {/* chips row */}
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                <RatingStars value={f.rating} onChange={() => {}} />

                <Chip
                  label={`Mood: ${(f?.entryId?.mood || "calm").toUpperCase()}`}
                  sx={{ fontWeight: 900 }}
                />

                <Chip
                  label={`Severity: ${(f?.entryId?.severity || "low").toUpperCase()}`}
                  sx={{
                    fontWeight: 900,
                    background: "rgba(15,23,42,0.05)",
                  }}
                />
              </Box>

              {/* comment */}
              {f.comment ? (
                <Typography sx={{ color: "text.secondary", fontSize: 13 }}>
                  “{f.comment}”
                </Typography>
              ) : null}
            </Box>
          ))}
        </Box>
      )}

      {/* ✅ Delete confirmation dialog */}
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontWeight: 1000 }}>Delete Feedback?</DialogTitle>

        <DialogContent sx={{ color: "text.secondary" }}>
          Are you sure you want to delete this feedback? This action cannot be undone.
          <Box sx={{ mt: 1, fontWeight: 900, color: "text.primary" }}>
            Rating: {selected?.rating} ⭐
          </Box>
          {selected?.comment ? (
            <Box sx={{ mt: 1 }}>
              “{selected.comment.slice(0, 120)}{selected.comment.length > 120 ? "..." : ""}”
            </Box>
          ) : null}
        </DialogContent>

        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setOpen(false)} sx={{ fontWeight: 900 }}>
            Cancel
          </Button>
          <Button
            onClick={confirmDelete}
            color="error"
            variant="contained"
            sx={{ fontWeight: 1000, borderRadius: 3 }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
