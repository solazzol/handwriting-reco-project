import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { TextField, Button, Box, Typography, Paper } from "@mui/material";

function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = () => {
    // Aggiungi la logica di autenticazione qui
    navigate("/home");
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        background: "linear-gradient(135deg, #3f51b5 50%, #f0f0f0 50%)",
      }}
    >
      <Paper
        elevation={6}
        sx={{
          padding: "30px",
          borderRadius: "12px",
          width: "400px",
          backgroundColor: "rgba(255, 255, 255, 0.8)",
        }}
      >
        <Typography variant="h4" textAlign="center" marginBottom={3} color="primary">
          Welcome
        </Typography>
        <TextField
          label="Email"
          variant="outlined"
          fullWidth
          margin="normal"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          sx={{ fontSize: "1.2rem" }}
        />
        <TextField
          label="Password"
          variant="outlined"
          fullWidth
          margin="normal"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          sx={{ fontSize: "1.2rem" }}
        />
        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleLogin}
          sx={{
            marginTop: 2,
            padding: "10px 0",
            fontSize: "1.1rem",
            borderRadius: "8px",
          }}
        >
          Sign in
        </Button>
      </Paper>
    </Box>
  );
}

export default LoginPage;
