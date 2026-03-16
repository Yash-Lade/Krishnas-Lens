import { useState } from "react";
import { Box, Typography, TextField, Button, Chip } from "@mui/material";
import RatingStars from "./RatingStars";
import useSnackbar from "../../hooks/useSnackbar";
import { createFeedbackApi } from "../../services/feedbackApi";

export default function FeedbackForm({ entryId }) {
  const { showSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);

  const [form, setForm] = useState({
    rating: 5,
    comment: "",
  });

  const submit = async () => {
    try {
      if (!entryId) {
        showSnackbar("Please submit at least 1 thought before giving feedback.", "warning");
        return;
      }

      setLoading(true);

      await createFeedbackApi({
        entryId,                 // ✅ REQUIRED
        rating: Number(form.rating),
        comment: form.comment?.trim() || "",
      });

      showSnackbar("Feedback submitted ✅", "success");
      setForm({ rating: 5, comment: "" });
    } catch (e) {
      const msg =
        e?.response?.data?.message ||
        e?.message ||
        "Feedback failed";

      showSnackbar(msg, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        borderRadius: 4,
        p: 2.5,
        background: "rgba(255,255,255,0.72)",
        border: "1px solid rgba(15,23,42,0.06)",
        backdropFilter: "blur(12px)",
        boxShadow: "0 14px 32px rgba(15,23,42,0.07)",
        display: "grid",
        gap: 2,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
        <Typography sx={{ fontWeight: 1000, fontSize: 18 }}>
          Feedback
        </Typography>

        <Chip
          label={entryId ? "Linked to your latest entry" : "Submit a thought first"}
          sx={{
            fontWeight: 900,
            borderRadius: 999,
            background: entryId ? "rgba(74,163,184,0.14)" : "rgba(239,68,68,0.10)",
          }}
        />
      </Box>

      {/* ✅ Subtitle */}
      <Typography sx={{ color: "text.secondary", fontSize: 13 }}>
        Help us improve Krishna’s Lens experience.
      </Typography>

      <Box>
        <Typography sx={{ fontWeight: 900, mb: 0.5 }}>
          Rating
        </Typography>
        <RatingStars
          value={form.rating}
          onChange={(v) => setForm((p) => ({ ...p, rating: v }))}
        />
      </Box>

      <TextField
        label="Comment (optional)"
        value={form.comment}
        onChange={(e) => setForm((p) => ({ ...p, comment: e.target.value }))}
        multiline
        minRows={4}
        placeholder="What did you like? What can we improve?"
        inputProps={{ maxLength: 300 }}
      />

      <Button
        disabled={loading || !entryId}
        onClick={submit}
        variant="contained"
        sx={{
          borderRadius: 4,
          fontWeight: 1000,
          py: 1.1,
          background: "linear-gradient(90deg, #4AA3B8, #2F8790)",
          boxShadow: "0 14px 28px rgba(74,163,184,0.22)",
        }}
      >
        {loading ? "Submitting..." : "Submit Feedback"}
      </Button>
    </Box>
  );
}
