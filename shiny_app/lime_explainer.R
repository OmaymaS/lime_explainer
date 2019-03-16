## The main code is by @christophM
## available at https://github.com/christophM/interpretable-ml-book/blob/master/manuscript/05.8-agnostic-lime.Rmd

## load libraries
library(shiny)
library(tidyverse)
library(iml)
library(shinythemes)

## import functions and set theme
source("utils.R")
theme_set(my_theme())
default_color = "azure4"

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
## predict training data labels
lime_training_df$predicted <-  predict(rf, lime_training_df)

# draw n_sample for the LIME explanations
df_sample = data.frame(x1 = rnorm(n_sample, x_means[1], x_sd[1]),
                       x2 = rnorm(n_sample, x_means[2], x_sd[2]))

# scale the samples
points_sample <-  apply(df_sample, 1, function(x){
  (x - x_means) / x_sd
}) %>% t

# calculate decision boundaries
grid_x1 <-  seq(from = lower_x1, to = upper_x1, length.out = n_grid)
grid_x2 <-  seq(from = lower_x2, to = upper_x2, length.out = n_grid)
grid_df <-  expand.grid(x1 = grid_x1, x2 = grid_x2)

x1_steps <-  unique(grid_df$x1)[seq(from = 1, to = n_grid, length.out = 20)]
x2_steps <-  unique(grid_df$x2)[seq(from = 1, to = n_grid, length.out = 20)]

## predict grid_df labels
grid_df$predicted <- predict(rf, newdata = grid_df) %>% 
  as.character() %>% 
  as.numeric()

# Define UI  ---------------------------------------------
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("LIME Explainer Demo"),
  sidebarLayout(
    sidebarPanel(
      
      ## select inputs for x1, x2
      sliderInput("explain_x1", "x1", min = -2, max = 2, value = 0.5, step = 0.1),
      sliderInput("explain_x2", "x2", min = -2, max = 1, value = -1, step = 0.1),
      
      ## check boks for showing labels/points
      checkboxInput("show_labels", "Show Labels", value = TRUE),
      checkboxInput("show_points", "Show Points", value = FALSE),
      
      helpText("Based on an example in:",
               tags$a("Interpretable Machine Learning", 
                      href = "https://christophm.github.io/interpretable-ml-book/lime.html"),
               "by",
               tags$a("Christoph Molnar.",
                      href = "https://github.com/christophM"))
      
    ),
    mainPanel(
      ## show plot
      plotOutput("scatter"),
      helpText("Original Model: Rendom Forest.",
               tags$br(),
               "Local Model: Logistic Regression."),
      verbatimTextOutput("point_selected")
    )
  )
)

# Define server logic ----------------------------------------------
server <- function(input, output) {
  
  ## df_explain
  df_explain <- reactive({
    df_ex <- data_frame(x1 = input$explain_x1,
                        x2 = input$explain_x2)
    df_ex$y_predicted = predict(rf, newdata = df_ex)[[1]]
    df_ex
  })
  
  output$point_selected = renderPrint({
    glue::glue("Selected point: {df_explain()$x1}, {df_explain()$x2}"
               # "\n",
               # "Predicted label: {df_explain()$y_predicted}"
               )
  })
  
  ## point_explain_scaled
  point_explain_scaled <- reactive({
    (c(input$explain_x1, input$explain_x2) - x_means) / x_sd
  })
  
  ## kernel width
  ## TODO: make it a variable
  kernel_width <-  sqrt(dim(df_sample)[2])*0.15
  
  ## df_sample
  df_sample_new <- reactive({
    df <- df_sample
    distances <- get_distances(point_explain_scaled(), 
                               points_sample = points_sample)
    
    df$weights <-  kernel(distances, kernel_width = kernel_width)
    df$predicted = predict(rf, newdata = df_sample)
    df
  })
  
  
  # fit a logistic regression model
  mod <- reactive({
    glm(predicted ~ x1 + x2,
        data = df_sample_new(),
        weights = df_sample_new()$weights,
        family='binomial')
  })
  
  ## predict the grid points labels
  grid_df_new <- reactive({
    df <- grid_df
    df$explained <-  predict(mod(), newdata = grid_df, type = 'response')
    df %>%
      mutate(explained_class = round(explained))
  })
  
  
  ## grid_df_small
  grid_df_small <-  reactive({
    grid_df_new() %>% 
      filter(grid_df$x1 %in% x1_steps,
             grid_df$x2 %in% x2_steps)
  })
  
  ## logistic_boundary_df
  logistic_boundary_df <- reactive({
    coefs <-  coefficients(mod())
    logistic_boundary_x1 <-  grid_x1
    logistic_boundary_x2 <-  - (1/coefs['x2']) * (coefs['(Intercept)'] + coefs['x1'] * grid_x1) 
    log_df <-  data_frame(x1 = logistic_boundary_x1,
                          x2 = logistic_boundary_x2) 
    
    log_df %>% 
      filter(x2 <= upper_x2, x2 >= lower_x2)
  })
  
  colors = c('#132B43', '#56B1F7')
  
  
  ## plot
  output$scatter <- renderPlot({
    g <- ggplot(grid_df_new())+
      geom_raster(aes(x = x1 , y = x2 , fill = predicted), alpha = 0.3, interpolate = TRUE)+
      geom_point(data = df_explain(),
                 aes(x = x1, y = x2), fill = 'red', shape = 21, size = 4, alpha = 0.6)+
      scale_x_continuous(limits = c(-2, 2)) +
      scale_y_continuous(limits = c(-2, 1)) +
      my_theme(legend.position = 'none')
    
    ## show labels
    if(input$show_labels){
      g <- g+
        geom_point(data = grid_df_small() %>% filter(explained_class == 1),
                   aes(x = x1, y = x2, color = explained), size = 5, shape = "+")+
        geom_point(data = grid_df_small() %>% filter(explained_class == 0),
                   aes(x = x1, y = x2, color = explained), size = 5, shape = "-")
    }
    
    ## show points
    if(input$show_points){
      g <- g+
        geom_point(data = df_sample_new(), aes(x = x1, y = x2, size = weights), alpha = 0.4)
    }
    
    ## add log reg boundary
    g <- g+
      geom_line(data = logistic_boundary_df(),
                aes(x = x1, y = x2), color = 'gray50')
    g
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)

