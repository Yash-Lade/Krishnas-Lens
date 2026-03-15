import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { Entry } from "../model/entry.model.js";

/**
 * ✅ Template-based Lens Generator (No ML)
 * Same thought ko 3 perspectives me convert karta hai.
 */
const generateLensViews = ({ thoughtText, mood, severity }) => {
  const text = (thoughtText || "").trim();

  // Emotional Lens
  const emotionalView = `I understand you’re feeling ${
    mood || "stressed"
  }. It’s completely okay to feel this way.  
What you’re experiencing is valid — and you’re not alone.  
Take a deep breath. One step at a time, you will get through this.`;

  // Strategic Lens
  const strategicView = `Let’s handle this situation step-by-step:

1) Identify the core issue: "${text.slice(0, 120)}${text.length > 120 ? "..." : ""}"
2) Write down 2–3 small actions you can take today.
3) Focus on what is in your control, not what is not.
4) If severity is "${
    severity || "Low"
  }", take a short break and then restart with a plan.
5) If required, talk to a trusted friend/mentor for guidance.

Small consistent steps = big results.`;

  // Spiritual Lens (Gita inspired, paraphrased)
  const spiritualView = `Krishna teaches *Samatvam* — balance in both success and failure.

You are not defined by a single moment.  
Do your best action (*karma*) with full sincerity, and let go of excessive worry about outcomes.

"Focus on duty, not the fruit." (Gita-inspired)

When the mind feels heavy, return to your center: calm breath, steady effort, and trust in the journey.`;

  return { emotionalView, strategicView, spiritualView };
};

/**
 * ✅ CREATE ENTRY (Submit Thought)
 * POST /api/entries
 * protected
 */
const createEntry = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { thoughtText, mood, severity } = req.body;

  if (!userId) throw new ApiError(401, "Unauthorized");

  if (!thoughtText || !thoughtText.trim()) {
    throw new ApiError(400, "Thought text is required");
  }

  if (thoughtText.trim().length < 10) {
    throw new ApiError(400, "Thought text must be at least 10 characters");
  }

  // ✅ generate lens outputs (no ML)
  const { emotionalView, strategicView, spiritualView } = generateLensViews({
    thoughtText,
    mood,
    severity,
  });

  const entry = await Entry.create({
    userId,
    thoughtText: thoughtText.trim(),
    mood: mood || "Other",
    severity: severity || "Low",
    emotionalView,
    strategicView,
    spiritualView,
  });

  return res
    .status(201)
    .json(new ApiResponse(201, entry, "Entry created successfully ✅"));
});

/**
 * ✅ GET MY ENTRIES (History)
 * GET /api/entries
 * protected
 */
const getMyEntries = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  if (!userId) throw new ApiError(401, "Unauthorized");

  const entries = await Entry.find({ userId }).sort({ createdAt: -1 });

  return res
    .status(200)
    .json(new ApiResponse(200, entries, "Entries fetched successfully ✅"));
});

/**
 * ✅ GET SINGLE ENTRY
 * GET /api/entries/:entryId
 * protected
 */
const getEntryById = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { entryId } = req.params;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!entryId) throw new ApiError(400, "Entry ID is required");

  const entry = await Entry.findOne({ _id: entryId, userId });

  if (!entry) throw new ApiError(404, "Entry not found");

  return res
    .status(200)
    .json(new ApiResponse(200, entry, "Entry fetched successfully ✅"));
});

/**
 * ✅ UPDATE ENTRY (Edit Thought)
 * PATCH /api/entries/:entryId
 * protected
 */
const updateEntry = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { entryId } = req.params;
  const { thoughtText, mood, severity } = req.body;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!entryId) throw new ApiError(400, "Entry ID is required");

  const entry = await Entry.findOne({ _id: entryId, userId });
  if (!entry) throw new ApiError(404, "Entry not found");

  // ✅ update fields if provided
  if (thoughtText && thoughtText.trim()) {
    if (thoughtText.trim().length < 10) {
      throw new ApiError(400, "Thought text must be at least 10 characters");
    }
    entry.thoughtText = thoughtText.trim();
  }

  if (mood) entry.mood = mood;
  if (severity) entry.severity = severity;

  // ✅ regenerate lens if any core field changed
  const { emotionalView, strategicView, spiritualView } = generateLensViews({
    thoughtText: entry.thoughtText,
    mood: entry.mood,
    severity: entry.severity,
  });

  entry.emotionalView = emotionalView;
  entry.strategicView = strategicView;
  entry.spiritualView = spiritualView;

  await entry.save();

  return res
    .status(200)
    .json(new ApiResponse(200, entry, "Entry updated successfully ✅"));
});

/**
 * ✅ DELETE ENTRY
 * DELETE /api/entries/:entryId
 * protected
 */
const deleteEntry = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { entryId } = req.params;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!entryId) throw new ApiError(400, "Entry ID is required");

  const entry = await Entry.findOneAndDelete({ _id: entryId, userId });

  if (!entry) throw new ApiError(404, "Entry not found");

  return res
    .status(200)
    .json(new ApiResponse(200, {}, "Entry deleted successfully ✅"));
});

export {
  createEntry,
  getMyEntries,
  getEntryById,
  updateEntry,
  deleteEntry,
};
