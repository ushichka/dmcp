# Paper Ratings
 - Weak Reject
 - Strong Accept
 - Borderline
 => Reject
 
# issues
## approach
 - manual feature matching is error-prone
 - outlier sensitivity of pose estimation
 - minimum measurement fits any set of measurements
 - why affine model is beeing fit
 - small reprojection errors not surprising (4 points provide 12 DOF)
 - Usually RANSAC is used
 - errors are large, median has variation
 - radial distortion not mentioned

## technical exposition
 - unstated conventions
 - conventions are buried in half-sentence
 - E for [R|t] can be confused with essential matrix
 - C same type of matrix as [R|t]
 - coordinate system relationships insufficiently explained (needs figure)
 - include scales for scenes
 - large errors hidden in log plots and robust statistics
 - experiment only on ushichka datasets
 - experiment numbers are hidden and hard to be found
 - Tables in sec 3.1 and 3.2 needed
 - where do we deviate from [6] exactly
 - include related work section
 - narrow niche
 - expand scientific quality and description
 - less words more data visualization putting the contributions in the spotlight

## meta
 - study concerns, unanswered questions
 - submit to more application oriented venue

