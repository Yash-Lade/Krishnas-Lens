import { Box, Typography, TextField, Button, Divider } from "@mui/material";

export default function ThoughtForm({ value, onChange, onSubmit, loading }) {
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