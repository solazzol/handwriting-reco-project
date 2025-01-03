import React, { useState } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  AppBar,
  Toolbar,
  IconButton,
  Divider,
  Drawer,
  createTheme,
  ThemeProvider,
} from "@mui/material";
import LogoutIcon from "@mui/icons-material/Logout";
import MenuIcon from "@mui/icons-material/Menu";
import { useNavigate } from "react-router-dom";

function HomePage() {
  const navigate = useNavigate();
  const [document, setDocument] = useState(null);
  const [recipients, setRecipients] = useState([]);
  const [recipientInput, setRecipientInput] = useState("");
  const [recipientError, setRecipientError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const [receivedDocuments, setReceivedDocuments] = useState([
    { id: 1, title: "Document 1", file: "document1.pdf", sender: "John Doe", date: "2023-12-01" },
    { id: 2, title: "Document 2", file: "document2.png", sender: "Jane Smith", date: "2023-12-05" },
  ]);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const theme = createTheme({
    palette: {
      primary: { main: "#0077B6" }, // Medium Blue
      secondary: { main: "#90E0EF" }, // Light Blue
      background: {
        default: "#CAF0F8", // Pale Blue
        paper: "#E3F2FD", // Slightly lighter paper background
      },
      text: {
        primary: "#023E8A", // Dark Blue for strong text
        secondary: "#0077B6", // Medium Blue for lighter text
      },
    },
    typography: {
      fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    },
  });

  const handleFileChange = (event) => {
    setDocument(event.target.files[0]);
    setDocumentError("");
  };

  const handleAddRecipient = () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(recipientInput)) {
      setRecipientError("Please enter a valid email address");
      return;
    }
    setRecipientError("");
    if (!recipients.includes(recipientInput)) {
      setRecipients([...recipients, recipientInput]);
      setRecipientInput("");
    }
  };

  const handleShareDocument = () => {
    if (!document) {
      setDocumentError("Please select a document to share");
      return;
    }
    if (recipients.length === 0) {
      setRecipientError("Please add at least one recipient");
      return;
    }
    setDocumentError("");
    setRecipientError("");
    console.log("Document shared:", document, recipients);
  };

  const handleSelectDocument = (doc) => {
    navigate(`/document/${doc.id}`);
  };

  const handleLogout = () => {
    navigate("/login");
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          display: "flex",
          height: "100vh",
          overflow: "hidden",
          background: "linear-gradient(135deg, #CAF0F8, #90E0EF)",
        }}
      >
        {/* Top Bar */}
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            background: "linear-gradient(135deg, #023E8A, #0077B6)",
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => setDrawerOpen(!drawerOpen)}
              sx={{ marginRight: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Document Sharing Platform
            </Typography>
            <IconButton color="inherit" onClick={handleLogout}>
              <LogoutIcon fontSize="large" />
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Left Drawer */}
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          sx={{
            [`& .MuiDrawer-paper`]: {
              width: 280,
              boxSizing: "border-box",
              backgroundColor: "#CAF0F8",
              boxShadow: 5,
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: "auto", padding: 3 }}>
            <Typography variant="h5" gutterBottom color="primary">
              Received Documents
            </Typography>
            <Divider />
            <List>
              {receivedDocuments.map((doc) => (
                <ListItem button key={doc.id} onClick={() => handleSelectDocument(doc)}>
                  <ListItemText
                    primary={doc.title}
                    secondary={`From: ${doc.sender}`}
                    sx={{ color: "text.primary" }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box
          sx={{
            flexGrow: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: 3,
            marginTop: "64px",
          }}
        >
          <Paper
            elevation={4}
            sx={{
              padding: 7,
              width: "600px",
              textAlign: "center",
              borderRadius: "16px",
              background: "white",
              boxShadow: "0px 8px 15px rgba(0, 0, 0, 0.1)",
            }}
          >
            <Typography variant="h5" gutterBottom color="primary">
              Share a Document
            </Typography>

            {/* Upload Document Section */}
            <Box
              sx={{
                marginBottom: 3,
                textAlign: "left",
              }}
            >
              <Typography variant="body2" gutterBottom color="text.secondary">
                Upload Document
              </Typography>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  border: "1px solid rgba(0, 0, 0, 0.23)",
                  borderRadius: "4px",
                  paddingX: 2,
                  paddingY: 1,
                  backgroundColor: "#F0F4F8",
                }}
              >
                <Button
                  variant="contained"
                  component="label"
                  sx={{
                    textTransform: "none",
                    background: "linear-gradient(45deg, #0077B6, #90E0EF)",
                    color: "white",
                    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)",
                    paddingX: 2,
                  }}
                >
                  Choose File
                  <input
                    type="file"
                    hidden
                    onChange={handleFileChange}
                    accept=".pdf,.jpg,.png"
                  />
                </Button>
                {document && (
                  <Typography
                    variant="body2"
                    sx={{
                      marginLeft: 2,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      maxWidth: "300px",
                      fontStyle: "italic",
                    }}
                  >
                    {document.name}
                  </Typography>
                )}
              </Box>
              {documentError && (
                <Typography variant="body2" color="error" sx={{ marginTop: 1 }}>
                  {documentError}
                </Typography>
              )}
            </Box>

            {/* Email Input */}
            <TextField
              label="Email"
              fullWidth
              margin="normal"
              value={recipientInput}
              onChange={(e) => setRecipientInput(e.target.value)}
              error={!!recipientError}
              helperText={recipientError}
            />
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, marginBottom: 2 }}>
              {recipients.map((recipient, index) => (
                <Button
                  key={index}
                  variant="outlined"
                  color="primary"
                  onClick={() => setRecipients(recipients.filter((r) => r !== recipient))}
                  sx={{
                    textTransform: "none",
                    fontSize: "0.9rem",
                  }}
                >
                  {recipient}
                </Button>
              ))}
            </Box>

            {/* Share Button */}
            <Button
              variant="contained"
              color="primary"
              onClick={handleShareDocument}
              sx={{
                borderRadius: "20px",
                paddingX: 3,
                marginTop: 2,
                background: "linear-gradient(45deg, #0077B6, #023E8A)",
                boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)",
              }}
            >
              Share
            </Button>
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default HomePage;
