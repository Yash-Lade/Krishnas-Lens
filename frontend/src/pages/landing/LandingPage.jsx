import React from "react";
import {
  Box,
  Container,
  Typography,
  Button,
  Stack,
  Paper,
  Divider,
} from "@mui/material";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import FavoriteRoundedIcon from "@mui/icons-material/FavoriteRounded";
import TrackChangesRoundedIcon from "@mui/icons-material/TrackChangesRounded";
import SpaRoundedIcon from "@mui/icons-material/SpaRounded";
import ChatBubbleOutlineRoundedIcon from "@mui/icons-material/ChatBubbleOutlineRounded";
import LightbulbOutlinedIcon from "@mui/icons-material/LightbulbOutlined";
import BookmarkAddedOutlinedIcon from "@mui/icons-material/BookmarkAddedOutlined";

import { useNavigate } from "react-router-dom";

import bgClouds from "../../assets/images/bg-watercolor-clouds.png";
import mandala from "../../assets/images/mandala-tile.png";
import heroKrishna from "../../assets/images/hero-krishna-feather.png";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: "100vh",
        position: "relative",
        overflowX: "hidden",
        background:
          "linear-gradient(180deg, #FBFDFF 0%, #EEF4FF 45%, #F3F1F8 100%)",
      }}
    >
      {/* Background watercolor */}
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          backgroundImage: `url(${bgClouds})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          opacity: 0.55,
          pointerEvents: "none",
        }}
      />

      {/* Mandala overlay */}
      <Box
        sx={{
          position: "absolute",
          top: { xs: "-180px", md: "-120px" },
          right: { xs: "-220px", md: "-180px" },
          width: { xs: 460, md: 720 },
          height: { xs: 460, md: 720 },
          backgroundImage: `url(${mandala})`,
          backgroundSize: "cover",
          opacity: 0.05,
          filter: "blur(0.2px)",
          pointerEvents: "none",
        }}
      />

      <Container
        maxWidth="lg"
        sx={{
          position: "relative",
          zIndex: 2,
          py: { xs: 2, md: 4 },
          px: { xs: 1.5, sm: 2 },
        }}
      >
        {/* NAVBAR */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
            px: { xs: 0.5, md: 2 },
            py: 1.5,
          }}
        >
          <Typography
            sx={{
              fontFamily: "Poppins",
              fontWeight: 700,
              fontSize: { xs: 18, md: 22 },
              color: "#2C5C70",
              letterSpacing: "-0.3px",
              cursor: "pointer",
            }}
            onClick={() => navigate("/")}
          >
            Krishna’s Lens
          </Typography>

          <Button
            variant="contained"
            endIcon={<ArrowForwardRoundedIcon />}
            onClick={() => navigate("/signup")}
            sx={{
              bgcolor: "#4AA3B8",
              px: { xs: 1.6, md: 2.4 },
              py: 1,
              borderRadius: 999,
              boxShadow: "0 12px 24px rgba(74,163,184,0.28)",
              transition: "all 0.25s ease",
              "&:hover": {
                bgcolor: "#2F8790",
                transform: "translateY(-1px)",
                boxShadow: "0 16px 30px rgba(74,163,184,0.35)",
              },
            }}
          >
            Get Started
          </Button>
        </Box>

        {/* HERO GLASS CARD */}
        <Paper
          elevation={0}
          sx={{
            mt: { xs: 2, md: 3.5 },
            overflow: "hidden",
            position: "relative",
            borderRadius: { xs: 5, md: 6 },
            border: "1px solid rgba(255,255,255,0.60)",
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.62), rgba(243,241,248,0.46))",
            backdropFilter: "blur(16px)",
            boxShadow: "0 18px 45px rgba(15, 23, 42, 0.10)",
          }}
        >
          {/* ✅ Premium glow layers (Dashboard-like) */}
          <Box
            sx={{
              position: "absolute",
              right: -170,
              top: -170,
              width: 540,
              height: 540,
              borderRadius: "50%",
              background: "rgba(74,163,184,0.24)",
              filter: "blur(120px)",
              opacity: 0.95,
              pointerEvents: "none",
            }}
          />
          <Box
            sx={{
              position: "absolute",
              left: -210,
              bottom: -230,
              width: 580,
              height: 580,
              borderRadius: "50%",
              background: "rgba(15,23,42,0.08)",
              filter: "blur(135px)",
              opacity: 0.55,
              pointerEvents: "none",
            }}
          />
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              background:
                "radial-gradient(circle at 70% 40%, rgba(74,163,184,0.16), transparent 58%)",
              filter: "blur(45px)",
              opacity: 0.95,
              pointerEvents: "none",
            }}
          />

          {/* ✅ CONTENT above glow */}
          <Box sx={{ position: "relative", zIndex: 2, p: { xs: 2.2, sm: 4, md: 6 } }}>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", md: "1.2fr 1fr" },
                gap: { xs: 3, md: 2 },
                alignItems: "center",
              }}
            >
              {/* Left content */}
              <Box>
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: 30, sm: 40, md: 52 },
                    lineHeight: { xs: "1.15", md: "1.08" },
                    color: "text.primary",
                    mb: 2,
                    wordBreak: "break-word",
                  }}
                >
                  Gain Insight and Clarity{" "}
                  <Box component="span" sx={{ color: "#4AA3B8" }}>
                    with Krishna’s Lens
                  </Box>
                </Typography>

                <Typography
                  sx={{
                    color: "text.secondary",
                    fontSize: { xs: 14.2, md: 16 },
                    maxWidth: 520,
                    mb: 3,
                  }}
                >
                  Let Krishna’s ancient wisdom guide you to balance, clarity,
                  and inner peace. Track your thoughts, understand your mood,
                  and improve your mindset.
                </Typography>

                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={1.5}
                  sx={{ width: { xs: "100%", sm: "auto" } }}
                >
                  <Button
                    fullWidth
                    variant="contained"
                    onClick={() => navigate("/signup")}
                    sx={{
                      bgcolor: "#4AA3B8",
                      py: 1.25,
                      px: 3.2,
                      fontWeight: 700,
                      boxShadow: "0 14px 28px rgba(74,163,184,0.28)",
                      transition: "all 0.25s ease",
                      "&:hover": {
                        bgcolor: "#2F8790",
                        transform: "translateY(-1px)",
                      },
                    }}
                  >
                    Get Started
                  </Button>

                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => navigate("/login")}
                    sx={{
                      py: 1.25,
                      px: 3.2,
                      fontWeight: 600,
                      color: "#2C5C70",
                      borderColor: "rgba(44,92,112,0.35)",
                      bgcolor: "rgba(255,255,255,0.45)",
                      transition: "all 0.25s ease",
                      "&:hover": {
                        borderColor: "#2F8790",
                        bgcolor: "rgba(255,255,255,0.65)",
                      },
                    }}
                  >
                    Try Demo
                  </Button>
                </Stack>
              </Box>

              {/* Right illustration */}
              <Box
                sx={{
                  position: "relative",
                  display: "flex",
                  justifyContent: { xs: "center", md: "flex-end" },
                }}
              >
                {/* ✅ Illustration glow boost */}
                <Box
                  sx={{
                    position: "absolute",
                    right: { xs: -60, md: -90 },
                    top: "50%",
                    transform: "translateY(-50%)",
                    width: { xs: 320, md: 420 },
                    height: { xs: 320, md: 420 },
                    borderRadius: "50%",
                    background: "rgba(74,163,184,0.22)",
                    filter: "blur(95px)",
                    opacity: 0.9,
                    pointerEvents: "none",
                  }}
                />

                <Box
                  component="img"
                  src={heroKrishna}
                  alt="Krishna flute illustration"
                  sx={{
                    position: "relative",
                    zIndex: 2,
                    width: { xs: "100%", sm: "82%", md: "100%" },
                    maxWidth: { xs: 380, md: 460 },
                    height: "auto",
                    opacity: 0.94,
                    filter:
                      "drop-shadow(0 18px 35px rgba(15,23,42,0.12))",
                    userSelect: "none",
                    pointerEvents: "none",
                  }}
                />
              </Box>
            </Box>

            {/* Quote Pill */}
            <Box sx={{ mt: { xs: 2.8, md: 4 } }}>
              <Paper
                elevation={0}
                sx={{
                  px: { xs: 2, sm: 3 },
                  py: { xs: 1.4, sm: 2 },
                  borderRadius: 999,
                  border: "1px solid rgba(255,255,255,0.55)",
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.60), rgba(243,241,248,0.45))",
                  backdropFilter: "blur(12px)",
                  boxShadow: "0 10px 22px rgba(15, 23, 42, 0.08)",
                  display: "flex",
                  alignItems: "center",
                  gap: 1.2,
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                }}
              >
                <Typography
                  sx={{
                    fontStyle: "italic",
                    color: "text.primary",
                    fontSize: { xs: 13.5, sm: 15 },
                    opacity: 0.9,
                  }}
                >
                  “You have the right to perform your duty, but not to the fruits
                  of action.”
                </Typography>

                <Typography sx={{ color: "text.secondary", fontSize: 13 }}>
                  — Bhagavad Gita 2:47 (paraphrased)
                </Typography>
              </Paper>
            </Box>
          </Box>

          <Divider sx={{ opacity: 0.35 }} />

          {/* Footer tiny */}
          <Box
            sx={{
              px: { xs: 2, md: 6 },
              py: 2,
              display: "flex",
              justifyContent: "space-between",
              gap: 2,
              flexWrap: "wrap",
              color: "text.secondary",
              fontSize: 13,
            }}
          >
            <Typography sx={{ fontSize: 13 }}>© 2026 Krishna’s Lens</Typography>
            <Stack direction="row" spacing={2}>
              <Typography sx={{ fontSize: 13 }}>Privacy Policy</Typography>
              <Typography sx={{ fontSize: 13 }}>Terms of Service</Typography>
            </Stack>
          </Box>
        </Paper>

        {/* ✅ SECTION: 3 LENS CARDS */}
        <SectionWrapper>
          <Typography
            sx={{
              fontFamily: "Poppins",
              fontWeight: 700,
              fontSize: { xs: 20, md: 24 },
              textAlign: "center",
              color: "#2C5C70",
              mb: 2.5,
            }}
          >
            Explore Three Lenses
          </Typography>

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                sm: "1fr 1fr",
                md: "1fr 1fr 1fr",
              },
              gap: 2,
            }}
          >
            <LensCard icon={<FavoriteRoundedIcon />} title="Emotional Lens" subtitle="Empathy & Support" />
            <LensCard icon={<TrackChangesRoundedIcon />} title="Strategic Lens" subtitle="Action Plan & Clarity" />
            <LensCard icon={<SpaRoundedIcon />} title="Spiritual Lens" subtitle="Wisdom & Calmness" />
          </Box>
        </SectionWrapper>

        {/* ✅ SECTION: HOW IT WORKS */}
        <SectionWrapper>
          <Typography
            sx={{
              fontFamily: "Poppins",
              fontWeight: 700,
              fontSize: { xs: 20, md: 24 },
              textAlign: "center",
              color: "#2C5C70",
              mb: 2.5,
            }}
          >
            How It Works
          </Typography>

          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr 1fr 1fr" },
              gap: 2,
            }}
          >
            <StepCard step="1" icon={<ChatBubbleOutlineRoundedIcon />} title="Share what’s on your mind" subtitle="Write your thought honestly." />
            <StepCard step="2" icon={<LightbulbOutlinedIcon />} title="Get 3 perspectives" subtitle="Emotional • Strategic • Spiritual" />
            <StepCard step="3" icon={<BookmarkAddedOutlinedIcon />} title="Save & reflect anytime" subtitle="History is always available." />
          </Box>
        </SectionWrapper>

        <Box sx={{ height: 40 }} />
      </Container>
    </Box>
  );
}

/* Reusable wrapper */
function SectionWrapper({ children }) {
  return (
    <Paper
      elevation={0}
      sx={{
        mt: { xs: 3, md: 4 },
        p: { xs: 2.2, md: 3 },
        borderRadius: 5,
        border: "1px solid rgba(255,255,255,0.55)",
        background:
          "linear-gradient(180deg, rgba(255,255,255,0.55), rgba(243,241,248,0.40))",
        backdropFilter: "blur(14px)",
        boxShadow: "0 14px 35px rgba(15,23,42,0.07)",
      }}
    >
      {children}
    </Paper>
  );
}

/* Lens Card */
function LensCard({ icon, title, subtitle }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2.2,
        borderRadius: 4,
        border: "1px solid rgba(255,255,255,0.55)",
        background: "rgba(255,255,255,0.50)",
        backdropFilter: "blur(10px)",
        boxShadow: "0 10px 18px rgba(15,23,42,0.06)",
        display: "flex",
        alignItems: "center",
        gap: 1.4,
      }}
    >
      <Box
        sx={{
          width: 44,
          height: 44,
          borderRadius: 3,
          display: "grid",
          placeItems: "center",
          bgcolor: "rgba(74,163,184,0.13)",
          color: "#2F8790",
        }}
      >
        {icon}
      </Box>

      <Box>
        <Typography sx={{ fontWeight: 800, fontSize: 14, color: "#213139" }}>
          {title}
        </Typography>
        <Typography sx={{ fontSize: 13, color: "text.secondary" }}>
          {subtitle}
        </Typography>
      </Box>
    </Paper>
  );
}

/* How it works */
function StepCard({ step, icon, title, subtitle }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2.4,
        borderRadius: 4,
        border: "1px solid rgba(255,255,255,0.55)",
        background: "rgba(255,255,255,0.50)",
        backdropFilter: "blur(10px)",
        boxShadow: "0 10px 18px rgba(15,23,42,0.06)",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.2, mb: 1.2 }}>
        <Box
          sx={{
            width: 34,
            height: 34,
            borderRadius: 999,
            display: "grid",
            placeItems: "center",
            bgcolor: "rgba(214,169,95,0.18)",
            color: "#D6A95F",
            fontWeight: 900,
          }}
        >
          {step}
        </Box>

        <Box
          sx={{
            width: 42,
            height: 42,
            borderRadius: 3,
            display: "grid",
            placeItems: "center",
            bgcolor: "rgba(74,163,184,0.13)",
            color: "#2F8790",
          }}
        >
          {icon}
        </Box>
      </Box>

      <Typography sx={{ fontWeight: 900, color: "#213139", mb: 0.4 }}>
        {title}
      </Typography>
      <Typography sx={{ color: "text.secondary", fontSize: 13 }}>
        {subtitle}
      </Typography>
    </Paper>
  );
}
