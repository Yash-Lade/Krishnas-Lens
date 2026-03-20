import axios from "axios";

export const analyzeThought = async (thoughtText) => {
  try {

    const response = await axios.post(
      "http://127.0.0.1:8000/analyze",
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