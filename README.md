# ModelSteelyard
Comparing Models by Statistical Method

**ModelSteelyard is the application of interval estimation and hypothesis testing in statistics to compare the effects of different machine learning, deep learning and reinforcement learning models.A contains interval estimates of the mean values of model evaluation indicators in different test sets,Interval Estimation of the Difference of the Mean Values of the Evaluation Indicators of the Two Models under the Same Test Conditions and Hypothesis test of model evaluation index comparison.**

1. mean_interval(values, conf_level=0.95)
**parameters**
values: array_like.Array containing numbers whose variance is desired. If a is not an array, a conversion is attempted.
conf_level=0.95: Confidence level
**returns**
A tuple contains the lower bound and upper bound of the confidence interval.

2. delta_interval(cls, control_group, test_group, alpha=0.5)
**parameters**
control_group: array_like.
test_group: array_like.
alpha: significance
**returns**
two float value: ower_bound, upper_bound

3. rise_interval(control_group, test_group)
**parameters**
control_group: array_like.
test_group: array_like.
**returns**
two float value: lower_rise, upper_rise

4. surpass_rate(control_group, test_group)
control_group: array_like.
test_group: array_like.
**returns**
float value: Probability of better control group
