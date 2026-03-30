import { Entry } from "../model/entry.model.js";
import { analyzeThought } from "../services/ai.service.js";

import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";

console.log("ENTRY CONTROLLER LOADED");


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
  let context = "life";

  let emotionalLens = "";
  let strategicLens = "";
  let spiritualLens = "";

  try {

    const analysis = await analyzeThought(thoughtText);

    console.log("PYTHON RESPONSE =>", analysis);

    emotion = analysis?.emotion || "neutral";
    context = analysis?.context || "life";

    emotionalLens =
      analysis?.perspectives?.emotional ||
      "Your feelings are acknowledged. Stay calm and reflect.";

    strategicLens =
      analysis?.perspectives?.strategic ||
      "Focus on actions within your control and move step by step.";

    spiritualLens =
      analysis?.perspectives?.spiritual ||
      "Krishna teaches balance in both success and failure.";

  } catch (error) {

    console.log("AI ERROR =>", error.message);

    emotionalLens =
      "Your feelings are acknowledged. Take a moment to breathe and reflect.";

    strategicLens =
      "Focus on actions within your control and move forward step by step.";

    spiritualLens =
      "Krishna teaches balance in both success and failure.";

  }

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

  const entry = await Entry.findOne({ _id: entryId, userId });

  if (!entry) throw new ApiError(404, "Entry not found");

  if (thoughtText && thoughtText.trim()) {

    entry.thoughtText = thoughtText.trim();

    const analysis = await analyzeThought(entry.thoughtText);

    entry.emotionalLens =
      analysis?.perspectives?.emotional || entry.emotionalLens;

    entry.strategicLens =
      analysis?.perspectives?.strategic || entry.strategicLens;

    entry.spiritualLens =
      analysis?.perspectives?.spiritual || entry.spiritualLens;

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