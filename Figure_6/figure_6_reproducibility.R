# Load necessary libraries
library(data.table)
library(igraph)
library(Rmisc)
library(dplyr)
library(ggplot2)
library(reshape2)
library(Matrix)
library(ggstance)
library(ggridges)
library(ggpubr)
library(scatterpie)
library(diptest)
library(gtable)

# Clear the workspace and trigger garbage collection
rm(list = ls())  # Remove all objects from the workspace
gc()  # Trigger garbage collection to free memory

# Define color palettes for the plots
color_values <- c("#282828", "#4F0906", "#8F100B", "#DB4742", "#CFDB00", "#4495DB", "#0E538F", "#082E4F")
binary_colors <- c("#7F00DB", "#01DB4E")

################################################################################################
############################################## CREATE 2016 PLOT ################################
################################################################################################

# Define file paths to ideology estimates and interaction matrix data
ideology_file = "Anon_CA_weighted_analysis_2016_n_link_3_reduced_infl_no_exleft.Rdata"
interaction_file = "Anon_weighted_interaction_matrix_2016_3_reduced_infl_no_exleft.Rdata"

# Load ideology estimates and interaction matrix from the specified files
load(ideology_file)
load(interaction_file)

# Load influencers data
influencers_data=fread("Anon_all_influencers_data_2016.csv", colClasses = "character")
influencers_data[,rank:=as.numeric(rank)]

top_5_cat=fread("Anon_all_top_5_cat_2016.csv", colClasses="character")
top_5_cat[,rank:=as.numeric(rank)]


# Assign numerical values to types for ideology scaling
influencers_data$n_type[influencers_data$type == "far-right"] <- 1
influencers_data$n_type[influencers_data$type == "right"] <- 0.66
influencers_data$n_type[influencers_data$type == "lean-right"] <- 0.33
influencers_data$n_type[influencers_data$type == "center"] <- 0
influencers_data$n_type[influencers_data$type == "lean-left"] <- -0.33
influencers_data$n_type[influencers_data$type == "left"] <- -0.66
influencers_data$n_type[influencers_data$type == "far-left"] <- -1
influencers_data$n_type[influencers_data$type == "fake"] <- 1.33

# Re-factor influencer type with readable labels
influencers_data$type <- factor(influencers_data$type,
                                levels = c("fake", "far-right", "right", "lean-right", "center",
                                           "lean-left", "left", "far-left"),
                                labels = c("Fake news", "Extreme bias right", "Right news", "Right leaning News",
                                           "Center news", "Left leaning news", "Left news", "Extreme bias left"))

# Create a rank table for top 5 categories and calculate color score for each influencer
top_5_cat_ranks <- as.data.frame.matrix(xtabs(1/rank ~ id + type, influencers_data))
top_5_cat_ranks$id <- rownames(top_5_cat_ranks)
# Calculate color score as a weighted sum of ideology values for each influencer
color_numeric_vector <- c(1.3, 0.99, 0.66, 0.33, 0, -0.33, -0.66, -0.99)
top_5_cat_ranks$color_score <- as.numeric(xtabs(1/rank ~ id + type, influencers_data) %*% color_numeric_vector / rowSums(xtabs(1/rank ~ id + type, influencers_data)))

# Reduce influencer data to unique entries with minimum rank
influencers_data <- influencers_data[, .(rank = min(rank),
                                         name = unique(name),
                                         type = .SD[rank == min(rank), type][1],
                                         n_type = .SD[rank == min(rank), n_type][1]),
                                     by = .(id)]

# Merge influencer data with ideology data
res_infl_data <- data.table(id = as.character(res$colnames), phi = as.numeric(scale(res$colcoord[, 1])))
found_influencers <- merge(influencers_data, res_infl_data, by = "id")

# Filter top 5 category ranks and merge with top influencer names
setDT(top_5_cat_ranks)
top_5_cat_ranks <- top_5_cat_ranks[id %in% top_5_cat$id]
top_5_cat_ranks <- merge(top_5_cat_ranks, top_5_cat[, .(name = unique(name)), by = .(id)], by = "id")

# Convert interaction matrix to sparse matrix
interaction_matrix <- Matrix(interaction_matrix, sparse = TRUE)
summ <- summary(interaction_matrix)

# Create a data.table from the sparse matrix
df <- data.table(user = rownames(interaction_matrix)[summ$i],
                 influencer = colnames(interaction_matrix)[summ$j],
                 Weight = summ$x)

# Merge influencer data with interaction data
df <- merge(df, influencers_data[, .(id, n_type, type)], by.x = "influencer", by.y = "id")

# Calculate user statistics (number of links and average leaning)
user_stat <- df[, .(n_of_links = .N, leaning = mean(n_type, na.rm = TRUE)), by = .(user)]
user_phi <- data.table(id = res$rownames, phi = as.numeric(scale(res$rowcoord[, 1])))
df <- merge(df, user_phi, by.x = "user", by.y = "id")

# Merge user statistics with latent ideology data
data_to_plot <- data.table(id = res$rownames, phi = as.numeric(scale(res$rowcoord[, 1])))
data_to_plot <- merge(data_to_plot, user_stat, by.x = "id", by.y = "user")

# Perform a correlation test between leaning and phi (latent ideology)
print(cor.test(data_to_plot[!is.na(leaning) & !is.na(phi), leaning],
               data_to_plot[!is.na(leaning) & !is.na(phi), phi], method = "pearson"))

# If correlation is negative, reverse the direction of phi
if (cor(data_to_plot[!is.na(leaning) & !is.na(phi), leaning],
        data_to_plot[!is.na(leaning) & !is.na(phi), phi], method = "pearson") < 0) {
  # Reverse phi for all relevant data tables
  data_to_plot$phi <- -1 * data_to_plot$phi
  df$phi <- -1 * df$phi
  found_influencers$phi <- -1 * found_influencers$phi
  res_infl_data$phi <- -1 * res_infl_data$phi
  user_phi$phi <- -1 * user_phi$phi
}

# Calculate influencer statistics for plotting
influencers_avg_audience <- df[, .(count = .N,
                                   mean_phi = mean(phi),
                                   med_phi = median(phi),
                                   min_val = min(phi),
                                   max_val = max(phi),
                                   q_1 = quantile(phi, probs = 0.25),
                                   q_3 = quantile(phi, probs = 0.75),
                                   sd_ph = sd(phi),
                                   up = quantile(phi, probs = 0.95),
                                   low = quantile(phi, probs = 0.05)),
                               by = .(id = influencer)]

# Calculate interquartile range (IQR) for influencers
influencers_avg_audience$iqr <- influencers_avg_audience$q_3 - influencers_avg_audience$q_1

# Merge with additional influencer data
influencers_avg_audience <- merge(influencers_avg_audience, influencers_data, by = "id")
influencers_avg_audience <- merge(influencers_avg_audience, res_infl_data[, .(id, infl_phi = phi)], by = "id")

# Prepare data for plotting top 5 influencers
top_5_data_to_plot <- influencers_avg_audience[id %in% top_5_cat$id]
top_5_data_to_plot <- top_5_data_to_plot[order(med_phi, decreasing = TRUE)]
top_5_data_to_plot$y <- seq(from = 0, to = 4, by = 4/nrow(top_5_data_to_plot))[1:nrow(top_5_data_to_plot)]

# Copy the data frame and assign a default type label "all"
df_to_plot = df
df_to_plot$type = "all"

# Combine the original data frame with the new one that has the "all" type
df_to_plot = rbind(df_to_plot, df)

### BOXPLOT STATS ####
# Filter data for the top 5 influencers and merge with position (y-coordinate) data for plotting
top_05_influencers_users = df[influencer %in% top_5_cat$id]
top_05_influencers_users = merge(top_05_influencers_users, top_5_data_to_plot[, .(id, y)], by.x = "influencer", by.y = "id")

### Add space for text labels in the plot ###
# Calculate the x-position for labels, adjusting based on the median ideology (med_phi)
top_5_data_to_plot$xpos = ifelse(top_5_data_to_plot$med_phi < 0, top_5_data_to_plot$low, top_5_data_to_plot$up)

# Further adjust x-position for better text placement
top_5_data_to_plot$xpos = ifelse(top_5_data_to_plot$med_phi < 0, top_5_data_to_plot$xpos - 0.1, top_5_data_to_plot$xpos + 0.1)

# Prepare data for pie charts by merging rank data with top 5 influencer position data
data_for_pies = top_5_cat_ranks
data_for_pies = merge(data_for_pies, top_5_data_to_plot[, .(id, y, xpos, med_phi)], by = "id")

# Convert to data table format for efficient processing
setDT(data_for_pies)


##### Distribution Statistics ####
# Combine user and influencer ideology distributions into a single data table
all_distribution = rbind(df[, .(phi = unique(phi), type = "users"), by = .(id = user)],
                         influencers_avg_audience[, .(phi = med_phi, type = "influencers"), by = .(id)])

# Convert to data table format for efficient processing
setDT(all_distribution)

####### Perform Dip Tests to check for multimodality ######
dip.test(all_distribution[type == "influencers", phi])  # Test for influencers' ideology distribution
dip.test(all_distribution[type == "users", phi])  # Test for users' ideology distribution

# Calculate boxplot statistics (count, quantiles) for top 5 influencers
dt_for_plot_bx = top_05_influencers_users[, .(count = .N,
                                              min = quantile(phi, 0.05),
                                              low = quantile(phi, 0.25),
                                              med = quantile(phi, 0.5),
                                              up = quantile(phi, 0.75),
                                              max = quantile(phi, 0.95)),
                                          by = .(y, type, n_type, influencer)]

# Create a boxplot with pie charts and text labels for top influencers
bp_2016 = ggplot() +
  # Add boxplot for each influencer
  geom_boxploth(data = dt_for_plot_bx,
                aes(y = y, xmin = min, xmax = max, xmiddle = med,
                    xlower = low, xupper = up,
                    color = type,
                    group = y),
                stat = "identity", show.legend = FALSE) +
  
  # Add text labels for influencer aliases on the left side (negative median ideology)
  geom_text(data = data_for_pies[med_phi < 0], aes(x = xpos - 0.1, y = y, label = name), color = "black", show.legend = FALSE, hjust = 1) +
  
  # Add text labels for influencer aliases on the right side (positive median ideology)
  geom_text(data = data_for_pies[med_phi > 0], aes(x = xpos + 0.1, y = y, label = name), color = "black", show.legend = FALSE, hjust = 0) +
  
  ## Add count labels next to the boxplots to show the number of users
  geom_text(data = dt_for_plot_bx[med < 0], aes(x = max + 0.1, y = y, label = paste0("n=", count)), color = "black", show.legend = FALSE, hjust = 0) +
  geom_text(data = dt_for_plot_bx[med > 0], aes(x = min - 0.1, y = y, label = paste0("n=", count)), color = "black", show.legend = FALSE, hjust = 1) +
  
  # Add pie charts representing the distribution of ideologies for top influencers
  geom_scatterpie(aes(x = xpos, y = y),
                  data = data_for_pies, cols = c("Fake news",
                                                 "Extreme bias right", "Right news", "Right leaning News",
                                                 "Center news",
                                                 "Left leaning news", "Left news", "Extreme bias left"), 
                  color = NA, 
                  pie_scale = 0.9,
                  alpha = 0.9) +
  
  # Set the theme and appearance for the plot
  theme_classic() +
  scale_x_continuous(limits = c(-1.8, 2.2)) +
  scale_color_manual(values = color_values) +
  scale_fill_manual(values = color_values, labels = c("fake news", "extreme bias right", "right news", "right leaning news", "center news",
                                                      "left leaning news", "left news", "extreme bias left")) +
  
  # Adjust the legend position and appearance
  theme(legend.position = c(0.5, 0.98),
        legend.direction = "horizontal",
        legend.background = element_rect(fill = NA, colour = NA),
        legend.title = element_blank(),
        axis.line.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        text = element_text(size = 14),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  
  labs(x = "Latent Ideology Scale") +
  guides(colour = guide_legend(nrow = 2,
                               keywidth = unit(1, "cm"),
                               keyheight = unit(1, "cm"),
                               override.aes = list(shape = 16,
                                                   size = 6)),
         fill = guide_legend(nrow = 2)) +
  coord_fixed()

###### DISTRIBUTION #####
# Compute density estimates for influencers and users
infl_dens = density(all_distribution[type == "influencers", phi], n = 2048, bw = "SJ")
usrs_dens = density(all_distribution[type == "users", phi], n = 2048, bw = "SJ")

# Combine the density estimates into a single data table for plotting
density_est = rbind(data.table(type = "influencers", x = infl_dens$x, y = infl_dens$y),
                    data.table(type = "users", x = usrs_dens$x, y = usrs_dens$y))

#### BARPLOT #### 
# Create a histogram to show the distribution of latent ideology for users and influencers
dens_2016 = ggplot(all_distribution, aes(x = phi, fill = type)) +
  geom_histogram(binwidth = 0.05, alpha = 0.5, aes(y = ..density..), position = "identity") +
  
  # Set the limits for the x-axis and y-axis
  xlim(-1.8, 2.2) +
  scale_y_continuous(breaks = seq(0, 9, by = 3), limits = c(0, 10)) +
  
  # Set the theme and appearance of the plot
  theme_classic() +
  scale_fill_manual(values = binary_colors) +
  theme(legend.title = element_blank(),
        legend.position = c(0.6, 0.7),
        text = element_text(size = 14),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  
  labs(x = "Latent ideology", y = "Density")


#####################################################################################
#####################################################################################
############################ CREATE 2020 PLOTS ######################################
#####################################################################################
#####################################################################################

# Set a numeric vector of color codes for various ideological categories
color_numeric_vector = c(0, 1.3, -0.99, 0.99, -0.33, 0.33, -0.66, 0.66)

# Define the paths to the ideology estimates and interaction matrix files
ideology_file = "Anon_CA_weighted_analysis_2020_n_link_3_reduced_infl_noexleft.Rdata"
interaction_file = "Anon_weighted_interaction_matrix_2020_3_reduced_infl_noexleft.Rdata"

#### LOAD IDEOLOGY ESTIMATES ####
load(ideology_file)  # Load the ideology estimates file

#### LOAD INTERACTION MATRIX ####
load(interaction_file)  # Load the interaction matrix file

# Read data tables of influencer information and top 5 categories
top_5_cat=fread("Anon_all_top_5_cat_2020.csv", colClasses = "character")
top_5_cat[,rank:=as.numeric(rank)]
influencers_data=fread("Anon_all_influencers_data_2020.csv", colClasses = "character")
influencers_data[,rank:=as.numeric(rank)]

# Assign numeric values to influencer types for further processing
influencers_data$n_type[influencers_data$type == "extreme_bias_right"] = 1
influencers_data$n_type[influencers_data$type == "right"] = 0.66
influencers_data$n_type[influencers_data$type == "lean_right"] = 0.33
influencers_data$n_type[influencers_data$type == "center"] = 0
influencers_data$n_type[influencers_data$type == "lean_left"] = -0.33
influencers_data$n_type[influencers_data$type == "left"] = -0.66
influencers_data$n_type[influencers_data$type == "extreme_bias_left"] = -1
influencers_data$n_type[influencers_data$type == "fake"] = 1.33

# Convert influencer types to factor variables with specific labels
influencers_data$type = factor(influencers_data$type, 
                               levels = c("fake", "extreme_bias_right", "right", "lean_right", "center",
                                          "lean_left", "left", "extreme_bias_left"),
                               labels = c("Fake news", "Extreme bias right", "Right news", "Right leaning News",
                                          "Center news", "Left leaning news", "Left news", "Extreme bias left"))

# Calculate rank-based statistics for top 5 influencers and prepare for plotting
top_5_cat_ranks = as.data.frame.matrix(xtabs(1/rank ~ id + type, influencers_data))
top_5_cat_ranks$id = rownames(top_5_cat_ranks)
### ADD CONTINUOUS COLOR FOR NAMES ####
color_numeric_vector = c(1.3, 0.99, 0.66, 0.33, 0, -0.33, -0.66, -0.99)
top_5_cat_ranks$color_score = as.numeric(xtabs(1/rank ~ id + type, influencers_data) %*% color_numeric_vector / rowSums(xtabs(1/rank ~ id + type, influencers_data)))

# Group influencer data by unique IDs and calculate minimum rank for each influencer
influencers_data = influencers_data[, .(rank = min(rank),
                                        name = unique(name),
                                        type = .SD[rank == min(rank), type][1],
                                        n_type = .SD[rank == min(rank), n_type][1]),
                                    by = .(id)]

####### CHECK THE ACCORDANCE BETWEEN PHI AND OUR CLASSIFICATION ####
# Merge influencer data with scaled phi values (first latent dimension)
res_infl_data = data.table(id = as.character(res$colnames), phi = as.numeric(scale(res$colcoord[, 1])))
found_influencers = merge(influencers_data, res_infl_data, by.x = "id", by.y = "id")

# Filter the top 5 categories and merge with influencer data
setDT(top_5_cat_ranks)
top_5_cat_ranks = top_5_cat_ranks[id %in% top_5_cat$id]
### ADD CONTINUOUS COLOR FOR NAMES ####
top_5_cat_ranks = merge(top_5_cat_ranks, top_5_cat[, .(name = unique(name)), by = .(id)], by = "id")

# Convert the interaction matrix to a sparse matrix and extract summary statistics
interaction_matrix = Matrix(interaction_matrix, sparse = TRUE)
summ = summary(interaction_matrix)

# Create a data table with user, influencer, and interaction weight information
df = data.table(user = rownames(interaction_matrix)[summ$i],
                influencer = colnames(interaction_matrix)[summ$j],
                Weight = summ$x)

# Merge influencer type and numeric type values into the data table
df = merge(df, influencers_data[, .(id, n_type, type)], by.x = "influencer", by.y = "id")

# Calculate user statistics based on the number of links and the leaning (average n_type)
user_stat = df[, .(n_of_links = .N, leaning = mean(n_type, na.rm = TRUE)), by = .(user)]
user_phi = data.table(id = res$rownames, phi = as.numeric(scale(res$rowcoord[, 1])))
df = merge(df, user_phi, by.x = "user", by.y = "id")

# Prepare data for plotting by merging user statistics with scaled phi values
data_to_plot = data.table(id = res$rownames, phi = as.numeric(scale(res$rowcoord[, 1])))
data_to_plot = merge(data_to_plot, user_stat, by.x = "id", by.y = "user")
print(cor(data_to_plot[!is.na(leaning) & !is.na(phi), leaning],
          data_to_plot[!is.na(leaning) & !is.na(phi), phi], method = "pearson"))

#### IF THEY ARE NOT ACCORDANT, CHANGE THE ORIENTATION OF IDEOLOGY ####
if (cor(data_to_plot[!is.na(leaning) & !is.na(phi), leaning],
        data_to_plot[!is.na(leaning) & !is.na(phi), phi], method = "pearson") < 0) {
  # Invert the sign of the phi values if correlation is negative
  data_to_plot$phi = -1 * data_to_plot$phi
  df$phi = -1 * df$phi
  found_influencers$phi = -1 * found_influencers$phi
  res_infl_data$phi = -1 * res_infl_data$phi
  user_phi$phi = -1 * user_phi$phi
}

#### CREATE STATS AND CALCULATE VALUES FOR PLOT NAMES ####
# Calculate summary statistics (mean, median, etc.) for influencers' audience ideology (phi)
influencers_avg_audience = df[, .(mean_phi = mean(phi), med_phi = median(phi),
                                  min_val = min(phi), max_val = max(phi),
                                  q_1 = quantile(phi, probs = 0.25),
                                  q_3 = quantile(phi, probs = 0.75),
                                  sd_ph = sd(phi),
                                  up = quantile(phi, probs = 0.95),
                                  low = quantile(phi, probs = 0.05)),
                              by = .(id = influencer)]

# Calculate the interquartile range (IQR) and merge additional influencer data
influencers_avg_audience$iqr = influencers_avg_audience$q_3 - influencers_avg_audience$q_1
influencers_avg_audience = merge(influencers_avg_audience, influencers_data, by.x = "id", by.y = "id")
influencers_avg_audience = merge(influencers_avg_audience, res_infl_data[, .(id, infl_phi = phi)], by.x = "id", by.y = "id")

# Prepare top 5 influencer data for plotting, including y positions
top_5_data_to_plot = influencers_avg_audience[id %in% top_5_cat$id]
top_5_data_to_plot = top_5_data_to_plot[order(med_phi, decreasing = TRUE)]
top_5_data_to_plot$y = seq(from = 0, to = 4, by = 4 / nrow(top_5_data_to_plot))[1:nrow(top_5_data_to_plot)]

# Create a copy of the dataframe 'df' for plotting
df_to_plot = df
df_to_plot$type = "all"  # Assign a new column 'type' with value 'all' for all rows
df_to_plot = rbind(df_to_plot, df)  # Duplicate the dataframe by appending it to itself

### BOXPLOT STATS ####
# Filter rows where the influencer is in the top 5 categories
top_05_influencers_users = df[influencer %in% top_5_cat$id]
# Merge the filtered data with the top 5 influencer positions (y coordinates) for plotting
top_05_influencers_users = merge(top_05_influencers_users, top_5_data_to_plot[, .(id, y)], by.x = "influencer", by.y = "id")

### Add space for text labels on the plot ###
# Calculate the x-position for text based on the median phi value
top_5_data_to_plot$xpos = ifelse(top_5_data_to_plot$med_phi < 0,
                                 top_5_data_to_plot$low,  # If median phi is negative, use the lower quantile
                                 top_5_data_to_plot$up)   # If positive, use the upper quantile

# Adjust x-position slightly to avoid overlap of text labels
top_5_data_to_plot$xpos = ifelse(top_5_data_to_plot$med_phi < 0,
                                 top_5_data_to_plot$xpos - 0.1,  # Shift left for negative phi
                                 top_5_data_to_plot$xpos + 0.1)  # Shift right for positive phi

# Prepare data for pie charts by merging top 5 ranks with plot positions
data_for_pies = top_5_cat_ranks
data_for_pies = merge(data_for_pies, top_5_data_to_plot[, .(id, y, xpos, med_phi)], by = "id")

# Convert data to a data.table object for efficient manipulation
setDT(data_for_pies)

##### DISTRIBUTION STATISTICS #####
# Combine user and influencer data for distribution analysis
all_distribution = rbind(df[, .(phi = unique(phi), type = "users"), by = .(id = user)],
                         influencers_avg_audience[, .(phi = med_phi, type = "influencers"), by = .(id)])

###### DIP TESTS FOR UNIMODALITY #####
# Perform dip tests (a test for unimodality) for users and influencers
dip.test(all_distribution[type == "users", phi])
dip.test(all_distribution[type == "influencers", phi])

# Calculate summary statistics (count, quantiles) for boxplot creation
dt_for_plot_bx = top_05_influencers_users[, .(count = .N,
                                              min = quantile(phi, 0.05),
                                              low = quantile(phi, 0.25),
                                              med = quantile(phi, 0.5),
                                              up = quantile(phi, 0.75),
                                              max = quantile(phi, 0.95)),
                                          by = .(y, type, n_type, influencer)]

# Create a boxplot using ggplot2
bp_2020 = ggplot() +
  geom_boxploth(data = dt_for_plot_bx,  # Boxplot for influencers
                aes(y = y, xmin = min, xmax = max, xmiddle = med,
                    xlower = low, xupper = up,
                    color = type,
                    group = y),
                stat = "identity", show.legend = FALSE) +
  geom_text(data = data_for_pies[med_phi < 0], aes(x = xpos - 0.1, y = y, label = name), color = "black", show.legend = FALSE, hjust = 1) +  # Text labels for left-leaning influencers
  geom_text(data = data_for_pies[med_phi > 0], aes(x = xpos + 0.1, y = y, label = name), color = "black", show.legend = FALSE, hjust = 0) +  # Text labels for right-leaning influencers
  ## Add count labels next to boxplots
  geom_text(data = dt_for_plot_bx[med < 0], aes(x = max + 0.1, y = y, label = paste0("n=", count)), color = "black", show.legend = FALSE, hjust = 0) +
  geom_text(data = dt_for_plot_bx[med > 0], aes(x = min - 0.1, y = y, label = paste0("n=", count)), color = "black", show.legend = FALSE, hjust = 1) +
  ### Add scatter pie charts to represent the distribution of types within each influencer's audience
  geom_scatterpie(aes(x = xpos, y = y),
                  data = data_for_pies, cols = c("Fake news",
                                                 "Extreme bias right", "Right news", "Right leaning News",
                                                 "Center news",
                                                 "Left leaning news", "Left news", "Extreme bias left"), 
                  color = NA, 
                  pie_scale = 0.9, 
                  alpha = 0.9) +
  theme_classic() +
  scale_x_continuous(limits = c(-1.8, 2.2)) +  # Set x-axis limits
  scale_color_manual(values = color_values) +  # Set custom colors for boxplot
  scale_fill_manual(values = color_values, labels = c("fake news", "extreme bias right", "right news", "right leaning news", "center news",
                                                      "left leaning news", "left news", "extreme bias left")) +  # Set custom fill colors and labels
  theme(legend.position = c(0.5, 0.98),
        legend.direction = "horizontal",
        legend.background = element_rect(fill = NA, colour = NA),
        legend.title = element_blank(),
        axis.line.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        text = element_text(size = 14),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  labs(x = "Latent Ideology Scale") +
  guides(colour = guide_legend(nrow = 2,
                               keywidth = unit(1, "cm"),
                               keyheight = unit(1, "cm"),
                               override.aes = list(shape = 16,
                                                   size = 6)),
         fill = guide_legend(nrow = 2)) +
  coord_fixed()

###### DISTRIBUTION PLOTS #####

# Calculate density estimates for influencers and users
infl_dens = density(all_distribution[type == "influencers", phi], n = 2048, bw = "SJ")
usrs_dens = density(all_distribution[type == "users", phi], n = 2048, bw = "SJ")

# Combine density estimates into a single data table for plotting
density_est = rbind(data.table(type = "influencers", x = infl_dens$x, y = infl_dens$y),
                    data.table(type = "users", x = usrs_dens$x, y = usrs_dens$y))

# Create a density histogram plot using ggplot2
dens_2020 = ggplot(all_distribution, aes(x = phi, fill = type)) +
  geom_histogram(binwidth = 0.05, alpha = 0.5, aes(y = ..density..), position = "identity") +  # Histogram of phi values
  xlim(-1.8, 2.2) +  # Set x-axis limits
  scale_y_continuous(breaks = seq(0, 9, by = 3), limits = c(0, 12)) +  # Set y-axis breaks and limits
  theme_classic() +
  scale_fill_manual(values = binary_colors) +  # Set fill colors for the histogram
  theme(legend.title = element_blank(),
        legend.position = c(0.5, 0.7),
        text = element_text(size = 14),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  labs(x = "Latent ideology", y = "Density")

# Arrange additional plots (e.g., from 2016 and 2020) into a grid with shared legend
figure = ggarrange(bp_2016, bp_2020, dens_2016, dens_2020, nrow = 2, ncol = 2, heights = c(2, 0.6), align = "hv", common.legend = TRUE)

# Save the final arranged figure as a PDF
ggsave("figure.pdf", figure, device = "pdf", path = ".", width = 17.34, height = 11.69, units = "in")
 
