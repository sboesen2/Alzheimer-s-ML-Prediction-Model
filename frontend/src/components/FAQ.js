import React, { useState } from 'react';
import { Typography, Container, Accordion, AccordionSummary, AccordionDetails, Box, Link, useTheme, Button } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SearchIcon from '@mui/icons-material/Search';
import InputAdornment from '@mui/material/InputAdornment';
import TextField from '@mui/material/TextField';

const faqData = [
  {
    question: "What is Alzheimer's disease?",
    answer: "Alzheimer's disease is a progressive neurodegenerative disorder that primarily affects memory and cognitive function. It's the most common cause of dementia in older adults. The disease is characterized by the buildup of abnormal proteins in the brain, leading to the death of brain cells and the gradual loss of mental functions."
  },
  {
    question: "How accurate is this prediction tool?",
    answer: "Our tool uses advanced machine learning algorithms trained on large-scale genetic studies. However, it's important to understand that it provides an estimate of genetic risk, not a diagnosis. The accuracy can vary depending on the specific genetic markers analyzed and the populations studied. Genetic risk is just one factor in Alzheimer's development - environmental and lifestyle factors also play significant roles. Always consult with healthcare professionals for a comprehensive evaluation."
  },
  {
    question: "What are the sources of data used in this tool?",
    answer: "Our tool utilizes data from multiple reputable sources to ensure comprehensive and accurate genetic risk assessment for Alzheimer's disease. Key data sources include UniProt, Proteomics DB, GWAS Catalog, and the STRING Database. I also incorporated data from various studies on specific Alzheimer's-related proteins and biomarkers. For a complete list and detailed description of our data sources, please visit our 'Data Sources' page."
  },
  {
    question: "Can this tool diagnose Alzheimer's disease?",
    answer: "No, this tool cannot diagnose Alzheimer's disease. It's designed to estimate genetic risk based on specific genetic markers. Alzheimer's diagnosis requires comprehensive clinical evaluation, including cognitive tests, brain imaging, and sometimes biomarker analysis. Our tool is for research and educational purposes only and should not replace professional medical advice or diagnosis."
  },
  {
    question: "How does genetics influence Alzheimer's risk?",
    answer: "Genetics plays a significant role in Alzheimer's disease risk. Certain genes, like APOE, can increase or decrease an individual's likelihood of developing the disease. Our tool analyzes specific genetic markers associated with Alzheimer's to provide a risk estimate based on these genetic factors."
  },
  {
    question: "What other factors besides genetics contribute to Alzheimer's risk?",
    answer: "While genetics is important, other factors also contribute to Alzheimer's risk. These include age (the primary risk factor), lifestyle factors (such as diet, exercise, and cognitive engagement), environmental factors, and other health conditions like cardiovascular disease and diabetes. Our tool focuses on genetic risk, but it's important to consider these other factors for a comprehensive understanding of Alzheimer's risk."
  },
  {
    question: "How often is this tool updated with new genetic research?",
    answer: "We strive to keep our tool up-to-date with the latest genetic research on Alzheimer's disease. Our team regularly reviews new studies and incorporates significant findings into our model. However, the field of Alzheimer's genetics is rapidly evolving, and it may take time for new discoveries to be validated and integrated into risk prediction models. We recommend checking our 'About' page for information on the most recent updates to our tool."
  },
  {
    question: "What should I do if the tool suggests I have a high genetic risk for Alzheimer's?",
    answer: "If the tool indicates a high genetic risk, it's important not to panic. Remember, genetic risk doesn't guarantee you'll develop Alzheimer's. Possible next steps: 1) Consult a healthcare professional or genetic counselor for a thorough evaluation. 2) Focus on modifiable risk factors: maintain a healthy diet, exercise regularly, engage in cognitive activities, and manage cardiovascular health. 3) Consider participating in Alzheimer's prevention studies or clinical trials. 4) Stay informed about the latest Alzheimer's research and prevention strategies."
  },
  {
    question: "Are there ways to prevent Alzheimer's disease?",
    answer: "While there's no guaranteed way to prevent Alzheimer's, research suggests several strategies that may reduce risk or delay onset: 1) Regular physical exercise, 2) Cognitive stimulation and lifelong learning, 3) Social engagement, 4) Heart-healthy diet (like the Mediterranean diet), 5) Quality sleep, 6) Stress management, 7) Controlling cardiovascular risk factors (blood pressure, cholesterol, diabetes). These lifestyle factors can contribute to overall brain health and potentially reduce Alzheimer's risk."
  },
];

function FAQ() {
  const [expanded, setExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [visibleFAQs, setVisibleFAQs] = useState(5);
  const theme = useTheme();

  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };

  const handleSearch = (event) => {
    setSearchTerm(event.target.value.toLowerCase());
    setVisibleFAQs(faqData.length); // Show all FAQs when searching
  };

  const filteredFAQs = faqData.filter(faq =>
    faq.question.toLowerCase().includes(searchTerm) ||
    faq.answer.toLowerCase().includes(searchTerm)
  );

  const showMoreFAQs = () => {
    setVisibleFAQs(prevVisible => Math.min(prevVisible + 5, faqData.length));
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom color="primary" align="center" sx={{ mb: 4 }}>
        Frequently Asked Questions
      </Typography>

      <TextField
        fullWidth
        variant="outlined"
        placeholder="Search FAQs..."
        onChange={handleSearch}
        sx={{ mb: 4 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon color="primary" />
            </InputAdornment>
          ),
        }}
      />

      {filteredFAQs.slice(0, visibleFAQs).map((faq, index) => (
        <Accordion
          key={index}
          expanded={expanded === `panel${index}`}
          onChange={handleChange(`panel${index}`)}
          sx={{
            backgroundColor: theme.palette.background.paper,
            '&:hover': {
              backgroundColor: theme.palette.action.hover,
            },
            mb: 2,
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon color="primary" />}
            aria-controls={`panel${index}bh-content`}
            id={`panel${index}bh-header`}
          >
            <Typography variant="h6" color="primary">{faq.question}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography>{faq.answer}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}

      {visibleFAQs < filteredFAQs.length && (
        <Box display="flex" justifyContent="center" mt={2}>
          <Button variant="outlined" color="primary" onClick={showMoreFAQs}>
            Show More FAQs
          </Button>
        </Box>
      )}

      <Box mt={4}>
        <Typography variant="body2" align="center">
          For more detailed information about our data sources, please visit our{' '}
          <Link href="/data-sources" color="primary">Data Sources</Link> page.
        </Typography>
      </Box>
    </Container>
  );
}

export default FAQ;