import axios from "axios";

export const analyzeThought = async (thoughtText) => {
  try {
    const mlApiUrl = process.env.ML_API_URL || "http://127.0.0.1:8000";
    const cleanMlApiUrl = mlApiUrl.endsWith("/") ? mlApiUrl.slice(0, -1) : mlApiUrl;

    const response = await axios.post(
      `${cleanMlApiUrl}/analyze`,
      {
        text: thoughtText
      }
    );

    const data = response.data;

    return {
      emotion: data?.emotion || "neutral",
      context: data?.context || "life",
      perspectives: data?.perspectives || {}
    };

  } catch (error) {

    console.log("PYTHON ML API ERROR =>", error.message);

    return {
      emotion: "neutral",
      context: "life",
      perspectives: {}
    };

  }
};