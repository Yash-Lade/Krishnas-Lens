import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";

import authRoutes from "./routes/user.routes.js";
import entryRoutes from "./routes/entry.routes.js";
import feedbackRoutes from "./routes/feedback.routes.js";

const app = express();

// ✅ Allowed origins
const allowedOrigins = [
  process.env.FRONTEND_URL,
  process.env.CORS_ORIGIN,
  "http://localhost:5173",
].filter(Boolean);

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin) return callback(null, true); // Postman
    if (allowedOrigins.includes(origin)) return callback(null, true);
    if (origin.includes(".vercel.app")) return callback(null, true);
    return callback(new Error("Not allowed by CORS"), false);
  },
  credentials: true,
  methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

// ✅ Middlewares
app.use(cors(corsOptions));
app.use(express.json({ limit: "16kb" }));
app.use(express.urlencoded({ extended: true, limit: "16kb" }));
app.use(express.static("public"));
app.use(cookieParser());

// ✅ Health check
app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "Krishna’s Lens Backend running ✅",
  });
});

// ✅ Debug route (IMPORTANT ✅)
// isko open karke check karna: http://localhost:5000/api/v1/test
app.get("/api/v1/test", (req, res) => {
  res.status(200).json({ success: true, message: "API base route working ✅" });
});

// ✅ Versioned API routes
app.use("/api/v1/users", authRoutes);
app.use("/api/v1/entries", entryRoutes);
app.use("/api/v1/feedback", feedbackRoutes);

// ✅ Global error handler
app.use((err, req, res, next) => {
  console.log("GLOBAL ERROR =>", err);

  const statusCode = err.statusCode || 500;
  const message = err.message || "Internal Server Error";

  return res.status(statusCode).json({
    success: false,
    message,
    errors: err.errors || [],
  });
});


export { app };
