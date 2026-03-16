import axiosInstance from "./axiosinstance";

export const registerUserApi = async (payload) => {
  const { data } = await axiosInstance.post("/users/register", payload);
  return data;
};

export const loginUserApi = async (payload) => {
  const { data } = await axiosInstance.post("/users/login", payload);
  return data;
};

export const logoutUserApi = async () => {
  const { data } = await axiosInstance.post("/users/logout");
  return data;
};

export const updateProfileApi = async (payload) => {
  const { data } = await axiosInstance.patch("/users/profile", payload);
  return data;
};
