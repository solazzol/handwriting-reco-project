import React, { useState } from "react";
import {
  Box,
  Typography,
  AppBar,
  Toolbar,
  IconButton,
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemText,
  createTheme,
  ThemeProvider,
} from "@mui/material";
import LogoutIcon from "@mui/icons-material/Logout";
import MenuIcon from "@mui/icons-material/Menu";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { useNavigate } from "react-router-dom";

function DocumentPage() {
  const navigate = useNavigate();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [receivedDocuments] = useState([
    { id: 1, title: "Document 1", file: "document1.pdf", sender: "John Doe", date: "2023-12-01" },
    { id: 2, title: "Document 2", file: "document2.png", sender: "Jane Smith", date: "2023-12-05" },
  ]);
  const [selectedDocument, setSelectedDocument] = useState(receivedDocuments[0]); // Example document pre-selected

  const handleLogout = () => {
    navigate("/login");
  };

  const handleBack = () => {
    navigate("/home");
  };

  const handleSelectDocument = (doc) => {
    setSelectedDocument(doc);
  };

  const theme = createTheme({
    palette: {
      primary: { main: "#023E8A" },
      secondary: { main: "#0077B6" },
      background: { default: "#CAF0F8" },
    },
    typography: {
      fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    },
  });

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: "flex", height: "100vh", overflow: "hidden", backgroundColor: theme.palette.background.default }}>
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
            <IconButton
              color="inherit"
              onClick={handleBack}
              sx={{ marginRight: 2 }}
            >
              <ArrowBackIcon fontSize="large" />
            </IconButton>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Document Viewer
            </Typography>
            <IconButton
              color="inherit"
              onClick={handleLogout}
              sx={{ fontSize: "1.5rem" }}
            >
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
              backgroundColor: "#90E0EF",
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: "auto", padding: 3 }}>
            <Typography variant="h5" gutterBottom>
              Received Documents
            </Typography>
            <Divider />
            <List>
              {receivedDocuments.map((doc) => (
                <ListItem button key={doc.id} onClick={() => handleSelectDocument(doc)}>
                  <ListItemText primary={doc.title} secondary={`From: ${doc.sender}`} />
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
          <Box
            sx={{
              padding: 4,
              maxWidth: "800px",
              width: "100%",
              textAlign: "center",
              boxShadow: 3,
              borderRadius: "16px",
              backgroundColor: "#FFFFFF",
            }}
          >
            <Typography variant="h4" gutterBottom color="primary">
              {selectedDocument.title}
            </Typography>
            <Typography variant="body1" gutterBottom>
              Sent by: {selectedDocument.sender}
            </Typography>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Date: {selectedDocument.date}
            </Typography>
            <Divider sx={{ marginY: 2 }} />
            <Typography variant="body1" sx={{ marginBottom: 2 }}>
              Here, you can implement logic to render the document content (e.g., PDF viewer, image display, or text content).
            </Typography>
            <Typography
              sx={{
                color: theme.palette.primary.main,
                cursor: "pointer",
                textDecoration: "underline",
              }}
              onClick={() => alert(`Opening file: ${selectedDocument.file}`)}
            >
              Open File
            </Typography>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default DocumentPage;
