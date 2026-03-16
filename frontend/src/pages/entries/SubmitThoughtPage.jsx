import { useState } from "react";
import { Box, Typography, Chip, Divider, Button } from "@mui/material";
import ThoughtForm from "../../components/entries/ThoughtForm";
import useSnackbar from "../../hooks/useSnackbar";
import { createEntryApi } from "../../services/entryApi";
import { useNavigate } from "react-router-dom";

export default function SubmitThoughtPage() {
  const { showSnackbar } = useSnackbar();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    thoughtText: "",
    mood: "calm",
    severity: "low",
  });

  const handleSubmit = async () => {
    try {
      setLoading(true);

      const res = await createEntryApi({
        thoughtText: form.thoughtText,
        mood: form.mood,
        severity: form.severity,
      });

      const entryId = res?.data?.data?._id || res?.data?._id;

      showSnackbar("Thought submitted successfully ✅", "success");
      navigate(`/entry/${entryId}`);
    } catch (err) {
      showSnackbar(err?.response?.data?.message || "Submit failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setForm({ thoughtText: "", mood: "calm", severity: "low" });
    showSnackbar("Cleared ✅", "info");
  };

  return (
    <Box sx={{ display: "grid", gap: 2 }}>
      {/* Header */}
      <Box>
        <Typography sx={{ color: "text.secondary" }}>
          Capture your emotions — clarity will follow.
        </Typography>
      </Box>

      {/* Main grid */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", lg: "1fr 360px" },
          gap: 2,
          alignItems: "start",
        }}
      >
        {/* Left: Form */}
        <ThoughtForm
          value={form}
          onChange={setForm}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {/* Right: Info panel */}
        <Box
          sx={{
            borderRadius: 4,
            p: 2.2,
            background: "rgba(255,255,255,0.72)",
            border: "1px solid rgba(15,23,42,0.06)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 14px 32px rgba(15,23,42,0.07)",
          }}
        >
          <Typography sx={{ fontWeight: 1000, mb: 1 }}>
            Quick Guidance
          </Typography>

          <Box sx={{ display: "grid", gap: 1 }}>
            <Chip
              label="✅ Write honestly, not perfectly"
              sx={{ justifyContent: "flex-start", fontWeight: 900 }}
            />
            <Chip
              label="✅ Keep it short and specific"
              sx={{ justifyContent: "flex-start", fontWeight: 900 }}
            />
            <Chip
              label="✅ Choose mood + severity carefully"
              sx={{ justifyContent: "flex-start", fontWeight: 900 }}
            />
          </Box>

          <Divider sx={{ my: 2, opacity: 0.6 }} />

          <Typography sx={{ fontWeight: 900, mb: 0.8 }}>
            Privacy
          </Typography>
          <Typography sx={{ color: "text.secondary", fontSize: 13, lineHeight: 1.7 }}>
            Your entries are visible only to your account. This is a safe space to
            reflect.
          </Typography>

          <Divider sx={{ my: 2, opacity: 0.6 }} />

          <Button
            onClick={clearForm}
            disabled={loading}
            fullWidth
            variant="outlined"
            sx={{ borderRadius: 4, fontWeight: 1000 }}
          >
            Clear Form
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
