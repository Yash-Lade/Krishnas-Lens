import { useMemo } from "react";
import {
  Box,
  Typography,
  TextField,
  MenuItem,
  Button,
  LinearProgress,
  Chip,
  Divider,
} from "@mui/material";

const moods = [
  { label: "Calm", value: "calm" },
  { label: "Happy", value: "happy" },
  { label: "Sad", value: "sad" },
  { label: "Angry", value: "angry" },
  { label: "Anxious", value: "anxious" },
  { label: "Confused", value: "confused" },
  { label: "Stressed", value: "stressed" },
  { label: "Tired", value: "tired" },
];

const severities = [
  { label: "Low", value: "low" },
  { label: "Medium", value: "medium" },
  { label: "High", value: "high" },
];

const severityPill = (s) => {
  if (s === "high") return { label: "High", bg: "rgba(239,68,68,0.14)", color: "#991B1B" };
  if (s === "medium") return { label: "Medium", bg: "rgba(234,179,8,0.16)", color: "#854D0E" };
  return { label: "Low", bg: "rgba(34,197,94,0.14)", color: "#166534" };
};

export default function ThoughtForm({ value, onChange, onSubmit, loading }) {
  const count = value.thoughtText?.trim()?.length || 0;

  const progress = useMemo(() => {
    const max = 300;
    return Math.min(100, Math.round((count / max) * 100));
  }, [count]);

  const sev = severityPill(value.severity);

  return (
    <Box
      sx={{
        borderRadius: 4,
        p: { xs: 2, md: 3 },
        background: "rgba(255,255,255,0.72)",
        border: "1px solid rgba(15,23,42,0.06)",
        backdropFilter: "blur(12px)",
        boxShadow: "0 16px 35px rgba(15,23,42,0.08)",
      }}
    >
      {loading && (
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{ mb: 2, borderRadius: 999 }}
        />
      )}

      {/* Pills row */}
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 2 }}>
        <Chip label={`Mood: ${value.mood?.toUpperCase()}`} sx={{ fontWeight: 1000 }} />
        <Chip
          label={`Severity: ${sev.label}`}
          sx={{ fontWeight: 1000, background: sev.bg, color: sev.color }}
        />
        <Chip label={`${count}/300`} sx={{ fontWeight: 1000 }} />
      </Box>

      <Typography sx={{ fontWeight: 1000, mb: 1 }}>
        What's on your mind?
      </Typography>

      <TextField
        value={value.thoughtText}
        onChange={(e) => onChange({ ...value, thoughtText: e.target.value })}
        fullWidth
        multiline
        minRows={7}
        placeholder="Write your thoughts here..."
        inputProps={{ maxLength: 300 }}
        sx={{
          "& .MuiOutlinedInput-root": {
            borderRadius: 4,
            background: "rgba(255,255,255,0.85)",
            "&:hover": { boxShadow: "0 10px 26px rgba(15,23,42,0.08)" },
            "&.Mui-focused": {
              boxShadow: "0 18px 40px rgba(74,163,184,0.16)",
            },
          },
        }}
      />

      <Divider sx={{ my: 2, opacity: 0.6 }} />

      {/* Two column dropdowns */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
          gap: 2,
          mb: 2,
        }}
      >
        <Box sx={{ display: "grid", gap: 1 }}>
          <Typography sx={{ fontWeight: 900 }}>Select your mood</Typography>
          <TextField
            select
            value={value.mood}
            onChange={(e) => onChange({ ...value, mood: e.target.value })}
            sx={{ "& .MuiOutlinedInput-root": { borderRadius: 4 } }}
          >
            {moods.map((m) => (
              <MenuItem key={m.value} value={m.value}>
                {m.label}
              </MenuItem>
            ))}
          </TextField>
        </Box>

        <Box sx={{ display: "grid", gap: 1 }}>
          <Typography sx={{ fontWeight: 900 }}>Select severity</Typography>
          <TextField
            select
            value={value.severity}
            onChange={(e) => onChange({ ...value, severity: e.target.value })}
            sx={{ "& .MuiOutlinedInput-root": { borderRadius: 4 } }}
          >
            {severities.map((s) => (
              <MenuItem key={s.value} value={s.value}>
                {s.label}
              </MenuItem>
            ))}
          </TextField>
        </Box>
      </Box>

      <Button
        disabled={loading || !value.thoughtText.trim()}
        onClick={onSubmit}
        fullWidth
        variant="contained"
        sx={{
          borderRadius: 4,
          py: 1.25,
          fontWeight: 1000,
          background: "linear-gradient(90deg, #4AA3B8, #2F8790)",
          boxShadow: "0 14px 28px rgba(74,163,184,0.22)",
        }}
      >
        Submit Thought
      </Button>

      <Typography
        variant="caption"
        sx={{
          mt: 1.2,
          textAlign: "center",
          color: "text.secondary",
          display: "block",
          fontWeight: 700,
        }}
      >
        “Balance is true mastery.” — Krishna’s Lens
      </Typography>
    </Box>
  );
}
