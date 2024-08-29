import React from 'react';
import { Box, Container, Typography, Link } from '@mui/material';

function Footer() {
  return (
    <Box component="footer" sx={{ bgcolor: 'background.paper', py: 6, mt: 'auto' }}>
      <Container maxWidth="lg">
        <Typography variant="body2" color="text.secondary" align="center">
          © {new Date().getFullYear()} Alzheimer's Risk Prediction. All rights reserved.
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          <Link color="inherit" href="https://www.linkedin.com/in/sam-boesen1" target="_blank" rel="noopener noreferrer">
            LinkedIn
          </Link>{' '}
          |{' '}
          <Link color="inherit" href="https://github.com/sboesen2" target="_blank" rel="noopener noreferrer">
            GitHub
          </Link>
        </Typography>
      </Container>
    </Box>
  );
}

export default Footer;