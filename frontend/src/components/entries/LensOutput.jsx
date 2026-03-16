import { Box, Typography } from "@mui/material";
import LensCard from "../common/LensCard";
import SpaOutlinedIcon from "@mui/icons-material/SpaOutlined";
import InsightsOutlinedIcon from "@mui/icons-material/InsightsOutlined";
import AutoAwesomeOutlinedIcon from "@mui/icons-material/AutoAwesomeOutlined";

export default function LensOutput({ data }) {
  const emotional = data?.emotionalLens || "Take a deep breath. Your feelings are valid.";
  const strategic = data?.strategicLens || "Break the situation into smaller steps and focus on one action.";
  const spiritual = data?.spiritualLens || "Let go of what you cannot control. Perform your duty with calm mind.";

  return (
    <Box sx={{ display: "grid", gap: 2 }}>
      <LensCard title="Emotional Lens" icon={<SpaOutlinedIcon />}>
        <Typography sx={{ color: "text.secondary" }}>{emotional}</Typography>
      </LensCard>

      <LensCard title="Strategic Lens" icon={<InsightsOutlinedIcon />}>
        <Typography sx={{ color: "text.secondary" }}>{strategic}</Typography>
      </LensCard>

      <LensCard title="Spiritual Lens" icon={<AutoAwesomeOutlinedIcon />}>
        <Typography sx={{ color: "text.secondary" }}>{spiritual}</Typography>
      </LensCard>
    </Box>
  );
}
