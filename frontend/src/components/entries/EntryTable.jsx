import { Box, IconButton, Typography, Tooltip } from "@mui/material";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import DeleteOutlineOutlinedIcon from "@mui/icons-material/DeleteOutlineOutlined";
import { timeAgo } from "../../services/timeAgo";

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
        return (
          <Box
            key={e._id}
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr 110px" },
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
              <Typography
                variant="caption"
                sx={{ color: "text.secondary", fontWeight: 700 }}
              >
                Submitted {timeAgo(e.createdAt)}
              </Typography>
            </Box>

            <Box
              sx={{
                display: "flex",
                justifyContent: { xs: "flex-start", md: "flex-end" },
                gap: 0.6,
              }}
            >
              <Tooltip title="View">
                <IconButton onClick={() => onView?.(e)} size="small">
                  <VisibilityOutlinedIcon fontSize="small" />
                </IconButton>
              </Tooltip>

              <Tooltip title="Delete">
                <IconButton
                  onClick={() => onDelete?.(e)}
                  size="small"
                  color="error"
                >
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