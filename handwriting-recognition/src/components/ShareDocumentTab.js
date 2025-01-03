import React, { useState } from "react";
import { TextField, Button, Box, Typography } from "@mui/material";

function ShareDocumentTab() {
  const [document, setDocument] = useState(null);
  const [recipient, setRecipient] = useState("");

  const handleUpload = (event) => {
    setDocument(event.target.files[0]);
  };

  const handleShare = () => {
    // Logica di condivisione del documento
    console.log("Documento condiviso con:", recipient);
  };

  return (
    <Box>
      <Typography variant="h6">Condividi un documento</Typography>
      <TextField
        label="Email del destinatario"
        variant="outlined"
        margin="normal"
        fullWidth
        value={recipient}
        onChange={(e) => setRecipient(e.target.value)}
      />
      <Button variant="contained" component="label">
        Carica Documento
        <input type="file" hidden onChange={handleUpload} />
      </Button>
      {document && <Typography>Documento selezionato: {document.name}</Typography>}
      <Button
        variant="contained"
        color="primary"
        onClick={handleShare}
        style={{ marginTop: "10px" }}
      >
        Condividi
      </Button>
    </Box>
  );
}

export default ShareDocumentTab;
