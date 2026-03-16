import { useEffect, useMemo, useState } from "react";
import { Box, Typography, Chip } from "@mui/material";
import { useNavigate } from "react-router-dom";

import Loader from "../../components/common/Loader";
import EntryTable from "../../components/entries/EntryTable";
import FeedbackForm from "../../components/feedback/FeedbackForm";
import FeedbackList from "../../components/feedback/FeedbackList";

import { getMyEntriesApi } from "../../services/entryApi";

// ✅ assets (use these only)
import featherGlow from "../../assets/images/feather-glow.png";

const moodUI = (mood) => {
  const map = {
    calm: { label: "Calm", bg: "rgba(59,130,246,0.12)", color: "#1D4ED8" },
    happy: { label: "Happy", bg: "rgba(34,197,94,0.14)", color: "#166534" },
    anxious: { label: "Anxious", bg: "rgba(234,179,8,0.16)", color: "#854D0E" },
    stressed: { label: "Stressed", bg: "rgba(234,179,8,0.16)", color: "#854D0E" },
    sad: { label: "Sad", bg: "rgba(148,163,184,0.22)", color: "#334155" },
  };

  return (
    map[mood] || {
      label: "Balanced",
      bg: "rgba(148,163,184,0.18)",
      color: "#334155",
    }
  );
};

const severityScore = (severity) => {
  if (severity === "high") return 85;
  if (severity === "medium") return 55;
  return 25;
};

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

  const topMood = recent?.[0]?.mood || "calm";
  const mood = moodUI(topMood);

  const score = useMemo(
    () => severityScore(recent?.[0]?.severity || "low"),
    [recent]
  );

  if (loading) return <Loader text="Loading dashboard..." />;

  return (
    <Box sx={{ display: "grid", gap: 2 }}>
      {/* ✅ Insight row */}
      <Box
        sx={{
          display: "grid",
          gap: 2,
          gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" },
        }}
      >
        {/* Mood */}
        <Box sx={{ ...glassCard, p: 2.2 }}>
          <Typography sx={{ fontWeight: 1000, fontSize: 14, color: "text.secondary" }}>
            Recent Mood
          </Typography>

          <Typography sx={{ fontWeight: 1000, fontSize: 22, mt: 0.4 }}>
            {mood.label}
          </Typography>

          <Chip
            label={recent.length ? "Tracked from last entry" : "No entries yet"}
            sx={{
              mt: 1.2,
              borderRadius: 999,
              fontWeight: 900,
              background: mood.bg,
              color: mood.color,
            }}
          />
        </Box>

        {/* Stress */}
        <Box sx={{ ...glassCard, p: 2.2 }}>
          <Typography sx={{ fontWeight: 1000, fontSize: 14, color: "text.secondary" }}>
            Stress Score
          </Typography>

          <Typography sx={{ fontWeight: 1000, fontSize: 22, mt: 0.4 }}>
            {score}/100
          </Typography>

          <Typography sx={{ mt: 1.1, color: "text.secondary" }}>
            Based on severity level
          </Typography>
        </Box>

        {/* Total */}
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

      {/* ✅ Quote Banner (FINAL aligned feather) */}
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
        {/* ✅ Background subtle aura */}
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
              label={mood.label}
              sx={{
                fontWeight: 1000,
                borderRadius: 999,
                px: 1,
                background: mood.bg,
                color: mood.color,
              }}
            />
            <Chip
              label={recent.length ? "Recent mood tracked" : "No entries yet"}
              sx={{
                fontWeight: 900,
                borderRadius: 999,
                background: "rgba(15,23,42,0.05)",
              }}
            />
          </Box>
        </Box>

        {/* RIGHT VISUAL (feather locked to corner) */}
        <Box
          sx={{
            zIndex: 1,
            position: "relative",
            height: "100%",
            minHeight: { xs: 120, md: 200 },
            display: { xs: "none", md: "block" },
          }}
        >
          {/* ✅ glass overlay gives premium depth */}
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              borderRadius: 4,
              background:
                "linear-gradient(180deg, rgba(255,255,255,0.40), rgba(255,255,255,0))",
            }}
          />

          {/* ✅ feather image perfectly aligned */}
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

          {/* ✅ extra soft highlight behind feather */}
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

      {/* ✅ Recent Entries */}
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

      {/* ✅ Feedback */}
      <Box sx={{ ...glassCard, p: { xs: 2, md: 2.5 } }}>
        <FeedbackForm entryId={latestEntryId} />
      </Box>

      <Box sx={{ ...glassCard, p: { xs: 2, md: 2.5 } }}>
        <FeedbackList limit={5} />
      </Box>
    </Box>
  );
}
