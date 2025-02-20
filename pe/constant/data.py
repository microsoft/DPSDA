#: The column name of the label ID
LABEL_ID_COLUMN_NAME = "PE.LABEL_ID"
#: The column name of the client ID (if using client-level DP)
CLIENT_ID_COLUMN_NAME = "PE.CLIENT_ID"

#: The column name of the clean histogram
CLEAN_HISTOGRAM_COLUMN_NAME = "PE.CLEAN_HISTOGRAM"
#: The column name of the DP histogram
DP_HISTOGRAM_COLUMN_NAME = "PE.DP_HISTOGRAM"
#: The column name of the post-processed (e.g., clipped) DP histogram
POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME = "PE.POST_PROCESSED_DP_HISTOGRAM"

#: The column name of the embedding
EMBEDDING_COLUMN_NAME = "PE.EMBEDDING"
#: The column name of the lookahead embedding
LOOKAHEAD_EMBEDDING_COLUMN_NAME = "PE.LOOKAHEAD_EMBEDDING"

#: The column name of the index of synthetic sample from the previous iteration that generates the current sample
PARENT_SYN_DATA_INDEX_COLUMN_NAME = "PE.PARENT_SYN_DATA_INDEX"
#: The column name of the flag that indicates whether the sample is from the last iteration
FROM_LAST_FLAG_COLUMN_NAME = "PE.FROM_LAST_FLAG"
#: The column name that indicates the fold ID of the variation API
VARIATION_API_FOLD_ID_COLUMN_NAME = "PE.VARIATION_API_FOLD_ID"

#: The column name of the image data
IMAGE_DATA_COLUMN_NAME = "PE.IMAGE"
#: The column name of the image label that is used for the model to generate the image
IMAGE_MODEL_LABEL_COLUMN_NAME = "PE.IMAGE_MODEL_LABEL"
#: The column name of the prompt for the image
IMAGE_PROMPT_COLUMN_NAME = "PE.IMAGE_PROMPT"

#: The column name of the text data
TEXT_DATA_COLUMN_NAME = "PE.TEXT"

#: The column name of the LLM request messages
LLM_REQUEST_MESSAGES_COLUMN_NAME = "PE.LLM.MESSAGES"
#: The column name of the LLM parameters
LLM_PARAMETERS_COLUMN_NAME = "PE.LLM.PARAMETERS"

#: The column name of the nearest neighbors voting IDs
HISTOGRAM_NEAREST_NEIGHBORS_VOTING_IDS_COLUMN_NAME = "PE.HISTOGRAM.NEAREST_NEIGHBORS.VOTING_IDS"
