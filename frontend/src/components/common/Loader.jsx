import { Box, CircularProgress, Typography } from "@mui/material";

export default function Loader({ text = "Loading..." }) {
  return (
    <Box
      sx={{
        minHeight: 220,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 2,
      }}
    >
      <CircularProgress />
      <Typography variant="body2" sx={{ color: "text.secondary" }}>
        {text}
      </Typography>
    </Box>
  );
}
