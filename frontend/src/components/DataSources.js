import React from 'react';
import { Typography, Container, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Link, Box } from '@mui/material';

const DataSources = () => {
  const dataSources = [
    {
      name: "UniProt",
      url: "https://www.uniprot.org/",
      dateAccessed: "July 29, 2024",
      dataExtracted: "Protein sequences and metadata for Abeta40 and Abeta42 variants",
      searchTerms: "Amyloid-beta, APP, Alzheimer's disease"
    },
    {
      name: "Proteomics DB",
      url: "https://www.proteomicsdb.org/",
      dateAccessed: "July 29, 2024",
      dataExtracted: "Protein sequences and metadata for Tau and APP protein",
      searchTerms: "APP, Tau"
    },
    {
      name: "GWAS Catalog",
      url: "https://www.ebi.ac.uk/gwas/efotraits/EFO_0006514",
      dateAccessed: "July 30, 2024",
      dataExtracted: "Alzheimer's disease biomarker measurement",
      searchTerms: "Alzheimer's disease biomarker measurement"
    },
    {
      name: "STRING",
      url: "https://string-db.org/",
      dateAccessed: "July 30, 2024",
      dataExtracted: "Human protein-protein physical interaction network data",
      searchTerms: "Not applicable (bulk download)",
      additionalInfo: "File: 9606.protein.physical.links.detailed.v12.0.txt.gz"
    }
  ];

  const additionalDataFiles = [
    { name: "PNF-tau measurements", size: "888 KB" },
    { name: "Soluble TREM2 measurements", size: "154 KB" },
    { name: "Cerebral amyloid deposition measurements", size: "52 KB" },
    { name: "Phosphorylated tau (p-tau) measurements", size: "37 KB" },
    { name: "Amyloid-beta measurements", size: "139 KB" },
    { name: "Family history of Alzheimer's disease", size: "362 KB" },
    { name: "Total tau (t-tau) measurements", size: "50 KB" },
    { name: "Alpha-synuclein data", size: "29 KB" },
    { name: "Alpha phase angle measurements", size: "168 KB" },
    { name: "T-tau/abeta amyloid-1-42 ratio measurements", size: "41 KB" },
    { name: "Amyloid plaque accumulation rate", size: "8 KB" },
    { name: "Plasma beta-amyloid-1-42 measurements", size: "8 KB" },
    { name: "Plasma beta-amyloid-1-40 measurements", size: "8 KB" },
    { name: "Protein biomarker associations with disease", size: "2,470 KB" },
    { name: "Protein Search Results - tau", size: "3 KB" },
    { name: "Protein Search Results - APP", size: "2 KB" },
    { name: "Cleaned Alzheimer's data", size: "377 KB" }
  ];

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ padding: '20px', marginTop: '20px' }}>
        <Typography variant="h4" gutterBottom>Data Sources</Typography>
        <Typography paragraph>
          Our Alzheimer's Risk Prediction Tool utilizes data from various reputable sources to provide accurate and comprehensive genetic risk assessments. Below is a detailed list of our primary data sources:
        </Typography>

        <TableContainer component={Paper} sx={{ marginBottom: '40px' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Source</TableCell>
                <TableCell>URL</TableCell>
                <TableCell>Date Accessed</TableCell>
                <TableCell>Data Extracted</TableCell>
                <TableCell>Search Terms</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataSources.map((source, index) => (
                <TableRow key={index}>
                  <TableCell>{source.name}</TableCell>
                  <TableCell>
                    <Link href={source.url} target="_blank" rel="noopener noreferrer">
                      {source.url}
                    </Link>
                  </TableCell>
                  <TableCell>{source.dateAccessed}</TableCell>
                  <TableCell>{source.dataExtracted}</TableCell>
                  <TableCell>{source.searchTerms}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        <Typography variant="h5" gutterBottom>Additional Data Files</Typography>
        <Typography paragraph>
          Our analysis also incorporates data from the following files:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {additionalDataFiles.map((file, index) => (
            <Paper key={index} sx={{ padding: '10px', minWidth: '200px', flex: '1 1 auto' }}>
              <Typography variant="subtitle1">{file.name}</Typography>
              <Typography variant="body2" color="text.secondary">File size: {file.size}</Typography>
            </Paper>
          ))}
        </Box>

        <Typography variant="h6" sx={{ marginTop: '40px', marginBottom: '10px' }}>Literature Citation</Typography>
        <Typography paragraph>
          For the STRING database:
        </Typography>
        <Typography paragraph sx={{ marginLeft: '20px', fontStyle: 'italic' }}>
          Szklarczyk, D., et al. (2021). The STRING database in 2021: customizable protein-protein networks, and functional characterization of user-uploaded gene/measurement sets. Nucleic Acids Research, 49(D1), D605–D612. https://doi.org/10.1093/nar/gkaa1074
        </Typography>

        <Typography variant="body2" sx={{ marginTop: '20px' }}>
          Note: This tool is regularly updated to incorporate the latest research findings. Last updated: August 17, 2024.
        </Typography>
      </Paper>
    </Container>
  );
};

export default DataSources;