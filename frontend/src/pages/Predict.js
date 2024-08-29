import React from 'react';
import PredictionForm from '../components/PredictionForm';

function Predict() {
  const handleSubmit = (formData) => {
    // Here you would typically make an API call to your Flask backend
    console.log('Form submitted with data:', formData);
    // You would then navigate to a results page or update state to show results
  };

  return (
    <div>
      <h2>Predict Alzheimer's Risk</h2>
      <PredictionForm onSubmit={handleSubmit} />
    </div>
  );
}

export default Predict;