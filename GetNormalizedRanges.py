import pandas as pd
from normalized import Normalize

# Step 1: Create a sample DataFrame
sample_data = pd.DataFrame({'polity2': range(-10, 11)})

# Step 2: Initialize and apply the Normalize class
normalizer = Normalize()
normalized_sample_data = normalizer.normalize(sample_data)

# Step 3: Combine the original and normalized data for comparison
combined_data = pd.concat([sample_data, pd.DataFrame(normalized_sample_data, columns=['normalized_polity2'])], axis=1)

# Display the combined data
print(combined_data)
