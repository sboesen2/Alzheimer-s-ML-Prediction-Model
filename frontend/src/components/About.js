import React from 'react';
import { Typography, Paper, Accordion, AccordionSummary, AccordionDetails, List, ListItem, ListItemText } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const About = () => {
  return (
    <Paper elevation={3} style={{ padding: '20px', margin: '20px 0' }}>
      <Typography variant="h4" gutterBottom>
        About Alzheimer's Risk Prediction Tool
      </Typography>
      <Typography paragraph>
        This tool uses advanced machine learning techniques to predict the risk of Alzheimer's disease based on genetic markers.
        It's designed for research purposes and should not be used as a diagnostic tool.
      </Typography>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Genetic Basis of Alzheimer's Disease</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Alzheimer's disease has a complex genetic basis. While some rare forms of early-onset Alzheimer's are caused by specific gene mutations,
            the more common late-onset Alzheimer's is influenced by multiple genes and their interactions with environmental factors.
          </Typography>
          <Typography paragraph>
            Key genes associated with Alzheimer's risk include:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="APOE (Apolipoprotein E)"
                secondary="The most significant known genetic risk factor. The ε4 allele of APOE increases risk, while the ε2 allele may be protective."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="TREM2 (Triggering Receptor Expressed on Myeloid cells 2)"
                secondary="Rare variants in this gene can significantly increase Alzheimer's risk."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="APP, PSEN1, and PSEN2"
                secondary="Mutations in these genes are associated with early-onset familial Alzheimer's disease."
              />
            </ListItem>
          </List>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Features Used in Our Model</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Our model uses several key features derived from genome-wide association studies (GWAS):
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="SNP-Risk Allele"
                secondary="Single Nucleotide Polymorphisms associated with increased Alzheimer's risk."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="P-Value"
                secondary="Statistical significance of the association between the SNP and Alzheimer's disease."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Odds Ratio (OR) or Beta"
                secondary="Measure of the effect size of the genetic variant on Alzheimer's risk."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Risk Allele Frequency"
                secondary="Prevalence of the risk-associated allele in the population."
              />
            </ListItem>
          </List>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Model Methodology</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            My risk prediction model uses a machine learning algorithm trained on data from large-scale GWAS studies.
            It integrates multiple genetic markers to provide a composite risk score.
          </Typography>
          <Typography paragraph>
            The model employs techniques such as:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Polygenic Risk Scoring"
                secondary="Combining the effects of multiple genetic variants."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="SHAP (SHapley Additive exPlanations) Values"
                secondary="For interpreting the contribution of each genetic factor to the overall risk prediction."
              />
            </ListItem>
          </List>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Limitations and Considerations</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            While this tool provides valuable insights into genetic risk factors, it's important to understand its limitations:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Not a Diagnostic Tool"
                secondary="This prediction does not diagnose Alzheimer's disease or guarantee its development."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Genetic Focus"
                secondary="The model doesn't account for environmental factors, lifestyle choices, or non-genetic risk factors."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Population Bias"
                secondary="The underlying data may not represent all ethnic and racial groups equally."
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Evolving Science"
                secondary="Our understanding of Alzheimer's genetics is constantly evolving, and new discoveries may impact risk assessments."
              />
            </ListItem>
          </List>
        </AccordionDetails>
      </Accordion>

      <Typography variant="h6" gutterBottom style={{ marginTop: '20px' }}>
        How to Use
      </Typography>
      <Typography paragraph>
        Input the required genetic data in the provided fields on the Predict page. The tool will calculate a risk
        score based on these inputs. This score indicates genetic predisposition based on the markers provided,
        not a definitive diagnosis or prognosis.
      </Typography>

      <Typography variant="body2" style={{ marginTop: '20px', fontStyle: 'italic' }}>
        Always consult with a healthcare professional or genetic counselor for interpretation of genetic information
        and medical advice. This tool is for research and educational purposes only.
      </Typography>
    </Paper>
  );
};

export default About;