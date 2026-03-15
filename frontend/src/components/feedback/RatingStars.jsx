import { Box, IconButton } from "@mui/material";
import StarRoundedIcon from "@mui/icons-material/StarRounded";

export default function RatingStars({ value = 0, onChange }) {
  return (
    <Box sx={{ display: "flex", gap: 0.3 }}>
      {[1, 2, 3, 4, 5].map((n) => (
        <IconButton
          key={n}
          size="small"
          onClick={() => onChange?.(n)}
          sx={{
            color: n <= value ? "#D6A95F" : "rgba(15,23,42,0.25)",
          }}
        >
          <StarRoundedIcon />
        </IconButton>
      ))}
    </Box>
  );
}
