import axios from "axios";

export const analyzeThought = async (text) => {
  try {

    console.log("➡️ Calling Python ML API...");

    const response = await axios.post(
      "http://127.0.0.1:8000/analyze",
      { text },
      { timeout: 5000 }
    );

    console.log("✅ ML RESPONSE:", response.data);

    return response.data;

  } catch (error) {

    console.log("❌ ML API ERROR:", error.message);

    return {
      emotion: "neutral",
      context: "general",
      verses: []
    };
  }
};