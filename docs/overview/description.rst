===============================================
Description
===============================================

The ML repo allows anyone to use machine-learning based automated classification.  This automated classification can work on both free text (essays, content, etc), and on numeric values.

Let's say that you have 10000 user reviews for 15 books (ie "I loved this!", "I didn't like it.", and so on). What you really want to do is use the user reviews to get an aggregate score for each book that indicates how well-received it is. But, in your haste to collect the data, you forgot to get scores from the users.  In this case, the text of the user reviews is your predictor, and the score that you want to collect from each user for each book is the target variable.

So, how do you turn the text into numbers?  One very straightforward way is to just label each of the reviews by hand on a scale from 0 (the user didn't like it at all) to 5 (they really loved it).  But, somewhere around review 200 you are going to start to get very sick of the whole process.  A less labor intensive way is to use automated classification.

If you choose to use automated classification for this task, you will score some reasonable subset of the reviews (if you score more, the classification will be more accurate, but 200 should be fine as a baseline).  Once you have your subset, which can also be called a "training" set, you will be able to "train" a machine learning model that learns how to map your scores to the text of the reviews.  It will then be able to automatically score the rest of the 9800 reviews.  Let's say you also want to take the user's activity level into account in order to weight the score.  You can add in a numeric predictor in addition to your existing text predictor (the review text itself) in order to predict the target variable (score).

This repo gives you a nice, clean way to do that via convenience functions grade, grade_generic, create, and create_generic.


