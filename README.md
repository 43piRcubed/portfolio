# portfolio
## To Clone any of these projects follow the steps below

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
# Create a directory on yur computer, so Git doesn't get messy, and go to it
mkdir my-dir 
cd my-dir

# Start a Git repository
git init

# Track repository, do not enter subdirectory
git remote add -f origin https://github.com/43piRcubed/portfolio

# Enable the tree check feature
git config core.sparseCheckout true

# Create a file in the path: .git/info/sparse-checkout
# That is inside the hidden .git directory that was created
# by running the command: git init
# And inside it enter the name of the sub directory you only want to clone
echo 'files' >> .git/info/sparse-checkout

## Download with pull, not clone
git pull origin master
```
