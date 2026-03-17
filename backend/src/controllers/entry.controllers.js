import { Entry } from "../model/entry.model.js";
import { analyzeThought } from "../services/ai.service.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";

console.log("ENTRY CONTROLLER LOADED");


/**
 * Generate Lens Views
 */
const generateLensViews = ({ thoughtText, mood, emotion, context, verses }) => {

  const text = (thoughtText || "").trim();

  const emotionalLens = `I sense that you may be feeling ${emotion || mood || "stressed"}.

Your feelings are valid. Moments of ${emotion || "difficulty"} are part of being human.

Pause for a moment. Breathe slowly. Even challenging emotions carry messages that can guide us toward growth.`;


  const strategicLens = `Let's approach this situation practically.

Context detected: ${context || "life situation"}

Step-by-step approach:
1. Identify the core concern: "${text.slice(0,120)}${text.length > 120 ? "..." : ""}"
2. Break the challenge into smaller actionable steps.
3. Focus on actions within your control.
4. Start with one small improvement today.
5. If needed, seek guidance from a mentor or trusted friend.

Progress is built through small consistent actions.`;


  const spiritualLens = `From a Bhagavad Gita perspective:

"${verses?.[0] || "You have a right to perform your duties, but not to the fruits of your actions."}"

Krishna teaches Samatvam — maintaining balance in both success and difficulty.

Focus on right action, stay calm, and trust the journey.`;

  return { emotionalLens, strategicLens, spiritualLens };
};



/**
 * CREATE ENTRY
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

  console.log("ENTRY CREATE CALLED");

  let emotion = "neutral";
  let context = "general";
  let verses = [];

  try {

    const analysis = await analyzeThought(thoughtText);

    console.log("ML RESPONSE =>", analysis);

    emotion = analysis?.emotion || emotion;
    context = analysis?.context || context;
    verses = analysis?.verses || [];

  } catch (error) {

    console.log("ML ERROR =>", error.message);

  }

  const { emotionalLens, strategicLens, spiritualLens } =
    generateLensViews({
      thoughtText,
      mood,
      emotion,
      context,
      verses
    });

  console.log("LENS GENERATED =>", {
    emotionalLens,
    strategicLens,
    spiritualLens
  });

  const entry = await Entry.create({
    userId,
    thoughtText: thoughtText.trim(),
    mood: mood || emotion || "calm",
    severity: severity || "low",
    emotionalLens,
    strategicLens,
    spiritualLens
  });

  return res
    .status(201)
    .json(new ApiResponse(201, entry, "Entry created successfully ✅"));

});



/**
 * GET MY ENTRIES
 */
const getMyEntries = asyncHandler(async (req, res) => {

  const userId = req.user?._id || req.user?.id;

  if (!userId) throw new ApiError(401, "Unauthorized");

  const entries = await Entry.find({ userId })
    .sort({ createdAt: -1 });

  return res
    .status(200)
    .json(new ApiResponse(200, entries, "Entries fetched successfully ✅"));
});



/**
 * GET ENTRY BY ID
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
 * UPDATE ENTRY
 */
const updateEntry = asyncHandler(async (req, res) => {

  const userId = req.user?._id || req.user?.id;
  const { entryId } = req.params;
  const { thoughtText, mood, severity } = req.body;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!entryId) throw new ApiError(400, "Entry ID is required");

  const entry = await Entry.findOne({ _id: entryId, userId });

  if (!entry) throw new ApiError(404, "Entry not found");

  if (thoughtText && thoughtText.trim()) {

    if (thoughtText.trim().length < 10) {
      throw new ApiError(400, "Thought text must be at least 10 characters");
    }

    entry.thoughtText = thoughtText.trim();

    const analysis = await analyzeThought(entry.thoughtText);

    const emotion = analysis?.emotion;
    const context = analysis?.context;
    const verses = analysis?.verses || [];

    const { emotionalLens, strategicLens, spiritualLens } =
      generateLensViews({
        thoughtText: entry.thoughtText,
        mood: mood || entry.mood,
        emotion,
        context,
        verses
      });

    entry.emotionalLens = emotionalLens;
    entry.strategicLens = strategicLens;
    entry.spiritualLens = spiritualLens;
  }

  if (mood) entry.mood = mood;
  if (severity) entry.severity = severity;

  await entry.save();

  return res
    .status(200)
    .json(new ApiResponse(200, entry, "Entry updated successfully ✅"));
});



/**
 * DELETE ENTRY
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
  deleteEntry
};