# GitHub Beginner Tutorial

This tutorial will guide you step-by-step through the basics of using GitHub. You'll learn how to:
1. Clone a repository to JupyterHub.
2. Create your own repository.
3. Push and pull changes.
4. Work with branches.

---

# 1. Clone a GitHub Repository to JupyterHub 
## 1.1 Clone the Entire Repository
### Instructions:
- Go to the repository you want to clone on GitHub.
- Click the **Code** button to get the HTTPS link.
- Open the terminal and run the following command.

### Command:
```bash
git clone https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges.git
```

- Alternatively, in JupyterHub:
  - Click on the Git tab in the right-hand sidebar.
  - Select Clone Repository and paste the HTTPS link into the prompt.


## 1.2 Clone Only a Specific Folder from a GitHub Repository
Instructions:
If you only want to clone a specific folder from the repository, you need to use sparse checkout. Follow these steps:

Open a terminal in JupyterHub by navigating to File -> New -> Terminal.
Create a temporary folder and initialize it as a Git repository

### Step 1: Create a project folder and navigate into it
```bash
(notebook) jovyan@jupyter-username:~$ mkdir project1
(notebook) jovyan@jupyter-username:~$ cd project1
```

### Step 2: Initialize a Git repository
Initialize an empty Git repository in the current directory.
```bash
(notebook) jovyan@jupyter-username:~/project1$ git init
```

### Step 3: Add a remote repository
Add a remote Git repository URL where your target repository is hosted.
```bash
(notebook) jovyan@jupyter-username:~/project1$ git remote add origin https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges.git
```

### Step 4: Initialize sparse checkout
Configure the repository to allow sparse checkout, meaning only specific folders/files will be downloaded.
```bash
(notebook) jovyan@jupyter-username:~/project1$ git sparse-checkout init --cone
```

### Step 5: Specify the folder for sparse checkout
Set the specific folder you want to download using the sparse checkout feature.
```bash
(notebook) jovyan@jupyter-username:~/project1$ git sparse-checkout set Project_Path
```

### Step 6: Pull the files from the remote repository
Pull the files from the remote repository, but only the specified folder will be downloaded.
```bash
(notebook) jovyan@jupyter-username:~/project1$ git pull origin main
```

### Note: The parent folder structure leading to this subfolder will still be present, as Git includes the necessary hierarchy for the selected folder.









---



# 2. Create a Repository
The team leader can create a repository for the team and invite team members. This is convenient for collaboration and sharing project files.



## Step 1: Log in to GitHub
1. Open a web browser and go to [GitHub](https://github.com).
2. Log in with your GitHub account credentials.



## Step 2: Navigate to the Repository Creation Page
1. In the top-right corner of the GitHub page, click the **+** icon.
2. Select **New repository** from the dropdown menu.



## Step 3: Set Up the Repository
1. **Choose a repository name**: Enter a unique name for your repository. For example, `my-new-repo`.
2. **Add a description (optional)**: Write a short description of your project.
3. **Select repository visibility**:
   - **Public**: Anyone can view your repository.
   - **Private**: Only you and collaborators can access it.
4. **Initialize the repository (optional)**:
   - Select **Add a README file** to include a basic README in the repository.
   - Optionally, add a **.gitignore template** to exclude specific files or a **license** for your project.



## Step 4: Create the Repository
1. Click the green **Create repository** button at the bottom of the page.
2. Your new repository is now created and ready to use.



## Step 5: Add Code to the Repository

### Option 1: If You Do Not Have an Existing Local Repository
1. Open the terminal.
2. Initialize a local Git repository and push it to GitHub:
   ```bash
   git init
   git remote add origin https://github.com/your-username/my-new-repo.git
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   
### Option 2: If You Already Have a Local Repository (e.g., Cloned from Another Repository, like what we did in section 1)

1. **Check the current remote repository**:
   ```bash
   git remote -v

2. **Remove the link to the original repository:**
   ```bash
   git remote remove origin

3. **Add the new repository as the remote:**

  ```bash
  git remote add origin https://github.com/your-username/New-Repository.git

4. **Verify the new remote:**

  ```bash
  git remote -v
  ```
  Ensure that the output shows the URL for new repository.















---






# 3. Push Code to a GitHub Repository
This guide explains how to push code from your local repository to a GitHub repository and handle the main branch properly.

## Step 1: Ensure You Have a Git Repository
1. Navigate to your project folder in the terminal.
2. Check if the folder is already a Git repository:
   ```bash
   git status
  ```
  If it’s not, initialize a Git repository:
  ```bash
  git init
  ```

### Step 2: Add Remote Repository
If you haven’t already linked your GitHub repository, add it as a remote:
```bash
git remote add origin https://github.com/your-username/your-repository.git
```
Verify the remote link:
```bash
git remote -v
```
### Step 3: Add and Commit Changes
Add all files to the staging area:
```bash
git add .
```
Commit the changes:

```bash
git commit -m "Your commit message here"
```

### Step 4: Push to GitHub
Push your code to the main branch of the remote repository:

```bash
git push -u origin main
```










---






# 4. Work with Branch
It allows you to work on different versions of a project simultaneously without interfering with the main codebase. Every Git repository starts with a default branch, typically called main.

To create a new branch:
```bash
git branch branch-name
```
To switch to the new branch:
```bash
git checkout branch-name
```
Or, create and switch to a new branch in one step:
```bash
git checkout -b branch-name
```

Merging Branches:

Once work in a branch is complete, you can merge it into the main branch:
```bash
git checkout main
git merge branch-name
```
Deleting Branches:

After merging, you can delete the branch to keep the repository clean:
```bash
git branch -d branch-name
```
Adding files and preparing to push to a branch is nearly identical to pushing to the main branch:

Stage and Commit Your Changes:
```bash
git add .
git commit -m "Your commit message here"
```
If it’s the first push for this branch (not yet created on the remote repository):
```bash
git push -u origin branch-name
```
The -u option sets origin/branch-name as the upstream branch for future pushes and pulls.
For subsequent pushes:
```bash
git push
```

---




# 5. Pull
The `git pull` command is used to fetch and merge changes from a remote repository into your local repository. It ensures your local branch stays up-to-date with changes made by others or on the remote.


Before pulling, ensure you are on the correct local branch where you want the updates:
```
git branch
```
The active branch is highlighted with an asterisk (*).

To pull updates from the remote branch into your current branch, use:
```bash
git pull origin branch-name
```
origin: Refers to the remote repository.

branch-name: The name of the branch you want to pull changes from (e.g., main or branch-name).


**Reference**:

https://docs.github.com/en/get-started

https://training.github.com/downloads/github-git-cheat-sheet.pdf


