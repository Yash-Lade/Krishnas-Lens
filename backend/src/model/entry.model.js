import mongoose from "mongoose";

const entrySchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },

    thoughtText: {
      type: String,
      required: true,
      trim: true,
      maxlength: 300,
    },

    mood: {
      type: String,
      enum: ["calm", "happy", "sad", "angry", "anxious", "confused", "stressed", "tired"],
      default: "calm",
      required: true,
    },

    severity: {
      type: String,
      enum: ["low", "medium", "high"],
      default: "low",
      required: true,
    },

    // optional: lens outputs
    emotionalLens: { type: String, default: "" },
    strategicLens: { type: String, default: "" },
    spiritualLens: { type: String, default: "" },
  },
  { timestamps: true }
);

export const Entry = mongoose.model("Entry", entrySchema);
