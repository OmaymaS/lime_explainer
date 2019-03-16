## load libraries
library(iml)
library(tidyverse)

source(here::here("./R/utils.R"))

set.seed(1)

# define range of set
lower_x1 <- -2
upper_x1 <- 2
lower_x2 <- -2
upper_x2 <- 1

## define parameters
n_training  <-  20000 # Size of the training set for the black box classifier
n_grid <-  100        # Size for the grid to plot the decision boundaries
n_sample <- 500        # Number of samples for LIME explanations

## define x1, x2
x1 <- runif(n_training, min = lower_x1, max = upper_x1)
x2 <-  runif(n_training, min = lower_x2, max = upper_x2)

## calculate mean and sd for scaling later
x_means = c(mean(x1), mean(x2))
x_sd = c(sd(x1), sd(x2))

# simulate y ~ x1 + x2
y <-  get_y(x1, x2)                          ## without noise
y_noisy <-  get_y(x1, x2, noise_prob = 0.01) ## add noise

## create lime df
lime_training_df <-  data.frame(x1 = x1,
                                x2 = x2,
                                y = as.factor(y),
                                y_noisy = as.factor(y_noisy))

## create rf model
rf <-  randomForest::randomForest(y_noisy ~ x1 + x2,
                                  data = lime_training_df, ntree = 100)

lime_training_df$predicted <-  predict(rf, lime_training_df)


## pick an observation to be explained
explain_x1 = 0.5
explain_x2 = -0.5

df_explain <- data.frame(x1 = explain_x1, x2 = explain_x2)
df_explain$y_predicted = predict(rf, newdata = df_explain)[[1]]

point_explain <-  c(explain_x1, explain_x2)
point_explain_scaled <-  (point_explain - x_means) / x_sd


# draw n_sample for the LIME explanations
df_sample = data.frame(x1 = rnorm(n_sample, x_means[1], x_sd[1]),
                       x2 = rnorm(n_sample, x_means[2], x_sd[2]))

# scale the samples
points_sample <-  apply(df_sample, 1, function(x){
  (x - x_means) / x_sd
}) %>% t

# add weights to the samples
kernel_width <-  sqrt(dim(df_sample)[2]) * 0.15

distances <-  get_distances(point_explain_scaled, 
                          points_sample = points_sample)

df_sample$weights <-  kernel(distances, kernel_width=kernel_width)

## predict on the sample
df_sample$predicted = predict(rf, newdata = df_sample)


# The decision boundaries
grid_x1 <-  seq(from = lower_x1, to = upper_x1, length.out = n_grid)
grid_x2 <-  seq(from = lower_x2, to = upper_x2, length.out = n_grid)
grid_df <-  expand.grid(x1 = grid_x1, x2 = grid_x2)

grid_df$predicted <- predict(rf, newdata = grid_df) %>% 
  as.character() %>% 
  as.numeric()


# fit a logistic regression model
mod <-  glm(predicted ~ x1 + x2,
            data = df_sample,
            weights = df_sample$weights,
            family='binomial')

## predict the grid points labels
grid_df$explained <-  predict(mod, newdata = grid_df, type = 'response')


## logistic decision boundary
coefs <-  coefficients(mod)
logistic_boundary_x1 <-  grid_x1
logistic_boundary_x2 <-  - (1/coefs['x2']) * (coefs['(Intercept)'] + coefs['x1'] * grid_x1) 
logistic_boundary_df <-  data.frame(x1 = logistic_boundary_x1, x2 = logistic_boundary_x2) 

logistic_boundary_df <- logistic_boundary_df %>% 
  filter(x2 <= upper_x2, x2 >= lower_x2)

## Visualization ---------------------------------------------------------------

# Create a smaller grid for visualization of local model boundaries
x1_steps <-  unique(grid_df$x1)[seq(from=1, to=n_grid, length.out = 20)]
x2_steps <-  unique(grid_df$x2)[seq(from=1, to=n_grid, length.out = 20)]

grid_df_small <-  grid_df %>% 
  filter(grid_df$x1 %in% x1_steps,
         grid_df$x2 %in% x2_steps) %>% 
  mutate(explained_class = round(explained))

colors = c('#132B43', '#56B1F7')


## data with some noise
p_data <-  ggplot(lime_training_df) +
  geom_point(aes(x = x1,y = x2,fill = y_noisy, color = y_noisy),
             alpha  = 0.3, shape = 21) +
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  my_theme(legend.position = 'none')


## decision boundaries of the learned black box classifier
p_boundaries <-  ggplot(grid_df) +
  geom_raster(aes(x = x1,y = x2,fill = predicted), alpha = 0.3, interpolate = TRUE) +
  my_theme(legend.position = 'none') +
  ggtitle('A')

## draw some samples
p_samples <-  p_boundaries +
  geom_point(data = df_sample, aes(x = x1, y = x2)) +
  scale_x_continuous(limits = c(-2, 2)) +
  scale_y_continuous(limits = c(-2, 1))


##point to be explained
p_explain <- p_samples +
  geom_point(data = df_explain, aes(x = x1,y = x2),
             fill = 'red', shape = 21, size = 4) +
  ggtitle('B')

## samples with weights
p_weighted <-  p_boundaries +
  geom_point(data = df_sample, aes(x = x1, y = x2, size = weights), alpha = 0.6) +
  scale_x_continuous(limits = c(-2, 2)) +
  scale_y_continuous(limits = c(-2, 1)) +
  geom_point(data = df_explain, aes(x = x1,y = x2),
             fill = 'red', shape = 21, size = 4) +
  ggtitle('C')

## log reg boundaries and labels
p_boundaries_lime <-  ggplot(grid_df)+
  geom_raster(aes(x = x1 , y = x2 , fill = predicted), alpha = 0.3, interpolate = TRUE)+
  geom_point(data = grid_df_small %>% filter(explained_class == 1),
             aes(x = x1, y = x2, color = explained), size = 2, shape = 3)+
  geom_point(data = grid_df_small %>% filter(explained_class == 0),
             aes(x = x1, y = x2, color = explained), size = 2, shape = 95)+
  geom_point(data = df_explain,
             aes(x = x1, y = x2), fill = 'red', shape = 21, size = 4, alpha = 0.6) +
  geom_line(data = logistic_boundary_df,
            aes(x = x1, y = x2), color = 'gray50')+
  # scale_x_continuous(limits = c(-2, 2)) +
  # scale_y_continuous(limits = c(-2, 1)) +
  my_theme(legend.position = 'none') + ggtitle('D')

gridExtra::grid.arrange(p_boundaries, p_explain, p_weighted, p_boundaries_lime, ncol = 2)