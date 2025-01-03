import React from "react";
import { List, ListItem, ListItemText, Typography, Box } from "@mui/material";

function ReceivedDocumentsTab() {
  const receivedDocuments = [
    { id: 1, name: "Documento 1", sender: "user1@example.com" },
    { id: 2, name: "Documento 2", sender: "user2@example.com" },
  ];

  return (
    <Box>
      <Typography variant="h6">Documenti ricevuti</Typography>
      <List>
        {receivedDocuments.map((doc) => (
          <ListItem key={doc.id}>
            <ListItemText
              primary={doc.name}
              secondary={`Da: ${doc.sender}`}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}

export default ReceivedDocumentsTab;
