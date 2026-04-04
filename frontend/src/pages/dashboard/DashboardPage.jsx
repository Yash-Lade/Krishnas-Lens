import { useEffect, useState } from "react";
import { Box, Typography, Chip } from "@mui/material";
import { useNavigate } from "react-router-dom";

import Loader from "../../components/common/Loader";
import EntryTable from "../../components/entries/EntryTable";
import FeedbackForm from "../../components/feedback/FeedbackForm";
import FeedbackList from "../../components/feedback/FeedbackList";

import { getMyEntriesApi } from "../../services/entryApi";

// assets
import featherGlow from "../../assets/images/feather-glow.png";

const glassCard = {
  borderRadius: 4,
  background: "rgba(255,255,255,0.72)",
  border: "1px solid rgba(15,23,42,0.06)",
  backdropFilter: "blur(12px)",
  boxShadow: "0 16px 35px rgba(15,23,42,0.08)",
};

export default function DashboardPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [entries, setEntries] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await getMyEntriesApi();
        setEntries(res?.data?.data || res?.data || []);
      } catch {
        setEntries([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const recent = entries.slice(0, 6);
  const latestEntryId = recent?.[0]?._id;

  if (loading) return <Loader text="Loading dashboard..." />;

  return (
    <Box sx={{ display: "grid", gap: 2 }}>
      {/* Insight row */}
      <Box
        sx={{
          display: "grid",
          gap: 2,
          gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" },
        }}
      >
        {/* Total Entries ONLY */}
        <Box sx={{ ...glassCard, p: 2.2 }}>
          <Typography sx={{ fontWeight: 1000, fontSize: 14, color: "text.secondary" }}>
            Total Entries
          </Typography>

          <Typography sx={{ fontWeight: 1000, fontSize: 22, mt: 0.4 }}>
            {entries.length}
          </Typography>

          <Typography sx={{ mt: 1.1, color: "text.secondary" }}>
            Reflections saved so far
          </Typography>
        </Box>
      </Box>

      {/* Quote Banner */}
      <Box
        sx={{
          ...glassCard,
          position: "relative",
          overflow: "hidden",
          minHeight: { xs: 190, md: 230 },
          p: { xs: 2.2, md: 3 },
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1.25fr 0.75fr" },
          alignItems: "center",
          gap: 2,
        }}
      >
        <Box
          sx={{
            position: "absolute",
            right: -120,
            top: -120,
            width: 340,
            height: 340,
            borderRadius: "50%",
            background: "rgba(74,163,184,0.20)",
            filter: "blur(95px)",
            opacity: 0.65,
            pointerEvents: "none",
          }}
        />
        <Box
          sx={{
            position: "absolute",
            left: -120,
            bottom: -160,
            width: 420,
            height: 420,
            borderRadius: "50%",
            background: "rgba(15,23,42,0.06)",
            filter: "blur(110px)",
            opacity: 0.55,
            pointerEvents: "none",
          }}
        />

        {/* LEFT TEXT */}
        <Box sx={{ zIndex: 2 }}>
          <Typography
            sx={{
              fontSize: { xs: 22, md: 30 },
              fontWeight: 1000,
              lineHeight: 1.18,
              maxWidth: 680,
            }}
          >
            Self-control is strength.
            <br />
            Right thought is mastery.
          </Typography>

          <Typography sx={{ mt: 1.1, color: "text.secondary", fontWeight: 900 }}>
            समत्वं योग उच्यते ..
          </Typography>

          <Typography sx={{ color: "text.secondary", maxWidth: 700 }}>
            Samatvam is yoga — balance is true mastery.
          </Typography>

          <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
            <Chip
              label={recent.length ? "Recent entries available" : "No entries yet"}
              sx={{
                fontWeight: 900,
                borderRadius: 999,
                background: "rgba(15,23,42,0.05)",
              }}
            />
          </Box>
        </Box>

        {/* RIGHT VISUAL */}
        <Box
          sx={{
            zIndex: 1,
            position: "relative",
            height: "100%",
            minHeight: { xs: 120, md: 200 },
            display: { xs: "none", md: "block" },
          }}
        >
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              borderRadius: 4,
              background:
                "linear-gradient(180deg, rgba(255,255,255,0.40), rgba(255,255,255,0))",
            }}
          />

          <Box
            component="img"
            src={featherGlow}
            alt="Feather glow"
            sx={{
              position: "absolute",
              right: -8,
              top: "50%",
              transform: "translateY(-50%) rotate(6deg)",
              width: 290,
              maxWidth: "100%",
              opacity: 0.98,
              pointerEvents: "none",
              filter: "drop-shadow(0 28px 55px rgba(0,0,0,0.12))",
            }}
          />

          <Box
            sx={{
              position: "absolute",
              right: -40,
              top: "50%",
              transform: "translateY(-50%)",
              width: 240,
              height: 240,
              borderRadius: "50%",
              background: "rgba(74,163,184,0.18)",
              filter: "blur(80px)",
              opacity: 0.75,
              pointerEvents: "none",
            }}
          />
        </Box>
      </Box>

      {/* Recent Entries */}
      <Box sx={{ ...glassCard, p: { xs: 2, md: 2.5 } }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: 2,
            gap: 2,
          }}
        >
          <Box>
            <Typography sx={{ fontWeight: 1000, fontSize: 18 }}>
              Recent Entries
            </Typography>
            <Typography sx={{ color: "text.secondary", fontSize: 13 }}>
              View your latest thoughts and insights.
            </Typography>
          </Box>

          <Chip
            onClick={() => navigate("/submit")}
            label="+ Submit a Thought"
            sx={{
              fontWeight: 1000,
              borderRadius: 999,
              cursor: "pointer",
              background: "rgba(74,163,184,0.12)",
              border: "1px solid rgba(74,163,184,0.20)",
              "&:hover": { background: "rgba(74,163,184,0.18)" },
            }}
          />
        </Box>

        <EntryTable
          rows={recent}
          onView={(e) => navigate(`/entry/${e._id}`)}
          onDelete={() => navigate("/history")}
        />
      </Box>

      {/* Feedback */}
      <Box sx={{ ...glassCard, p: { xs: 2, md: 2.5 } }}>
        <FeedbackForm entryId={latestEntryId} />
      </Box>

      <Box sx={{ ...glassCard, p: { xs: 2, md: 2.5 } }}>
        <FeedbackList limit={5} />
      </Box>
    </Box>
  );
}