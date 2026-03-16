import { Box, Chip, IconButton, Typography, Tooltip } from "@mui/material";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";
import { timeAgo } from "../../services/timeAgo";

const moodStyle = (mood) => {
  const map = {
    calm: { bg: "rgba(59,130,246,0.12)", color: "#1D4ED8", label: "CALM" },
    happy: { bg: "rgba(34,197,94,0.14)", color: "#166534", label: "HAPPY" },
    sad: { bg: "rgba(148,163,184,0.22)", color: "#334155", label: "SAD" },
    angry: { bg: "rgba(239,68,68,0.14)", color: "#991B1B", label: "ANGRY" },
    anxious: { bg: "rgba(234,179,8,0.16)", color: "#854D0E", label: "ANXIOUS" },
    stressed: { bg: "rgba(234,179,8,0.16)", color: "#854D0E", label: "STRESSED" },
  };
  return map[mood] || { bg: "rgba(148,163,184,0.18)", color: "#334155", label: "BALANCED" };
};

const severityStyle = (sev) => {
  const map = {
    low: { bg: "rgba(34,197,94,0.14)", color: "#166534", label: "LOW" },
    medium: { bg: "rgba(234,179,8,0.16)", color: "#854D0E", label: "MED" },
    high: { bg: "rgba(239,68,68,0.14)", color: "#991B1B", label: "HIGH" },
  };
  return map[sev] || map.low;
};

export default function EntryTable({ rows = [], onView, onDelete }) {
  if (!rows.length) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography sx={{ color: "text.secondary" }}>No entries yet.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: "grid", gap: 1 }}>
      {rows.map((e) => {
        const mood = moodStyle(e.mood);
        const sev = severityStyle(e.severity);

        return (
          <Box
            key={e._id}
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr 150px 110px 110px" },
              gap: 1,
              alignItems: "center",
              p: 1.25,
              borderRadius: 3,
              background: "rgba(255,255,255,0.72)",
              border: "1px solid rgba(15,23,42,0.06)",
              transition: "0.18s",
              "&:hover": {
                background: "rgba(255,255,255,0.92)",
                boxShadow: "0 14px 28px rgba(15,23,42,0.08)",
              },
            }}
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ fontWeight: 900 }} noWrap>
                {e.thoughtText}
              </Typography>
              <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 700 }}>
                Submitted {timeAgo(e.createdAt)}
              </Typography>
            </Box>

            <Chip
              label={mood.label}
              sx={{
                width: "fit-content",
                borderRadius: 999,
                fontWeight: 1000,
                background: mood.bg,
                color: mood.color,
              }}
            />

            <Chip
              label={sev.label}
              sx={{
                width: "fit-content",
                borderRadius: 999,
                fontWeight: 1000,
                background: sev.bg,
                color: sev.color,
              }}
            />

            <Box sx={{ display: "flex", justifyContent: { xs: "flex-start", md: "flex-end" }, gap: 0.6 }}>
              <Tooltip title="View">
                <IconButton onClick={() => onView?.(e)} size="small">
                  <VisibilityOutlinedIcon fontSize="small" />
                </IconButton>
              </Tooltip>

              <Tooltip title="Delete">
                <IconButton onClick={() => onDelete?.(e)} size="small" color="error">
                  <DeleteOutlineOutlinedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}
