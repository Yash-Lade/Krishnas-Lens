import { asyncHandler } from "../utils/asyncHandler.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { Feedback } from "../model/feedback.model.js";
import { Entry } from "../model/entry.model.js";

/**
 * ✅ CREATE FEEDBACK
 * POST /api/feedback
 * protected
 */
const createFeedback = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { entryId, rating, comment } = req.body;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!entryId) throw new ApiError(400, "Entry ID is required");
  if (!rating) throw new ApiError(400, "Rating is required");

  const numericRating = Number(rating);
  if (Number.isNaN(numericRating) || numericRating < 1 || numericRating > 5) {
    throw new ApiError(400, "Rating must be between 1 and 5");
  }

  // ✅ ensure entry exists and belongs to current user
  const entry = await Entry.findOne({ _id: entryId, userId });
  if (!entry) throw new ApiError(404, "Entry not found");

  // ✅ prevent duplicate feedback on same entry by same user
  const alreadyGiven = await Feedback.findOne({ entryId, userId });
  if (alreadyGiven) {
    throw new ApiError(409, "Feedback already submitted for this entry");
  }

  const feedback = await Feedback.create({
    entryId,
    userId,
    rating: numericRating,
    comment: comment?.trim() || "",
  });

  return res
    .status(201)
    .json(new ApiResponse(201, feedback, "Feedback submitted successfully ✅"));
});

/**
 * ✅ GET MY FEEDBACK (optional)
 * GET /api/feedback/my
 * protected
 */
const getMyFeedback = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  if (!userId) throw new ApiError(401, "Unauthorized");

  const feedbacks = await Feedback.find({ userId })
    .populate("entryId", "thoughtText mood severity createdAt")
    .sort({ createdAt: -1 });

  return res
    .status(200)
    .json(new ApiResponse(200, feedbacks, "Feedback fetched successfully ✅"));
});

/**
 * ✅ DELETE FEEDBACK (optional)
 * DELETE /api/feedback/:feedbackId
 * protected
 */
const deleteFeedback = asyncHandler(async (req, res) => {
  const userId = req.user?._id || req.user?.id;
  const { feedbackId } = req.params;

  if (!userId) throw new ApiError(401, "Unauthorized");
  if (!feedbackId) throw new ApiError(400, "Feedback ID is required");

  const deleted = await Feedback.findOneAndDelete({ _id: feedbackId, userId });

  if (!deleted) throw new ApiError(404, "Feedback not found");

  return res
    .status(200)
    .json(new ApiResponse(200, {}, "Feedback deleted successfully ✅"));
});

export { createFeedback, getMyFeedback, deleteFeedback };
