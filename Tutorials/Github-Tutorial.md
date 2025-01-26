# GitHub Beginner Tutorial

This tutorial will guide you step-by-step through the basics of using GitHub. There are **two methods** for performing Git operations:
1. **Using the Terminal**: Ideal for users who prefer command-line control and advanced flexibility.
2. **Using the JupyterHub Git Interface**: A user-friendly interface for performing common Git tasks visually.

You will learn how to:
- Clone a repository to JupyterHub.
- Create your own repository.
- Push and pull changes.
- Work with branches.

---



# 01. Authenticate with a GitHub Personal Access Token (PAT)

To access private repositories or when GitHub requires additional security (like: pushing code to GitHub from the terminal for the first time), you will need to use a Personal Access Token instead of your GitHub password.

## Step 1: Generate a Personal Access Token

1. Log in to GitHub.

2. Click on your profile picture in the upper-right corner and select Settings.

3. Navigate to Developer settings > Personal access tokens > Tokens (classic).

4. Click on Generate new token.

5. Select the scopes you need (e.g., repo for full repository access).

6. Copy the generated token and store it securely. 

## Step 2: Use the Token for Authentication

When prompted for your GitHub credentials:

- Enter your GitHub username as usual.

- Instead of your password, paste the Personal Access Token.

If using HTTPS to clone a repository, you can include the token directly in the URL:

```bash
https://<your-username>:<your-token>@github.com/<owner>/<repository>.git
```


---

# 02. GitHub Basics: Essential Operations
## 2.1: Using the Terminal

### 2.1.1 Clone a Repository to JupyterHub
1. Open the terminal in JupyterHub (`File -> New -> Terminal`).
2. Go to the GitHub repository you want to clone, click **Code**, and copy the HTTPS link.
3. Use the following command to clone the repository:
   ```bash
   git clone https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges.git
   ```



### 2.1.2 Create a Repository
The team leader can create a repository for the team and invite team members. This is convenient for collaboration and sharing project files.



#### Step 1: Log in to GitHub
1. Open a web browser and go to [GitHub](https://github.com).
2. Log in with your GitHub account credentials.



#### Step 2: Navigate to the Repository Creation Page
1. In the top-right corner of the GitHub page, click the **+** icon.
2. Select **New repository** from the dropdown menu.



#### Step 3: Set Up the Repository
1. **Choose a repository name**: Enter a unique name for your repository. For example, `my-new-repo`.
2. **Add a description (optional)**: Write a short description of your project.
3. **Select repository visibility**:
   - **Public**: Anyone can view your repository.
   - **Private**: Only you and collaborators can access it.
4. **Initialize the repository (optional)**:
   - Select **Add a README file** to include a basic README in the repository.
   - Optionally, add a **.gitignore template** to exclude specific files or a **license** for your project.



#### Step 4: Create the Repository
1. Click the green **Create repository** button at the bottom of the page.
2. Your new repository is now created and ready to use.



#### Step 5: Add Code to the Repository

- Case 1: If You Do Not Have an Existing Local Repository
    1. Open the terminal.
    2. Initialize a local Git repository and push it to GitHub:
       ```bash
       git init
       git remote add origin https://github.com/your-username/my-new-repo.git
       git add .
       git commit -m "Initial commit"
       git push -u origin main
      ```
   
- Case 2: If You Already Have a Local Repository (e.g., Cloned from Another Repository, like what we did in section 1)
    
    1. **Check the current remote repository**:
       ```bash
       git remote -v
       ```
    
    2. **Remove the link to the original repository:**
       ```bash
       git remote remove origin
    
    3. **Add the new repository as the remote:**
      ```bash
      git remote add origin https://github.com/your-username/New-Repository.git
      ```
    
    4. **Verify the new remote:**
      ```bash
      git remote -v
      ```


  Ensure that the output shows the URL for new repository.






### 2.1.3 Push Code to a GitHub Repository
This guide explains how to push code from your local repository to a GitHub repository and handle the main branch properly.

#### Step 1: Ensure You Have a Git Repository
Navigate to your project folder in the terminal. Check if the folder is already a Git repository:
```bash
git status
```
If it’s not, initialize a Git repository:

```bash
git init
```

#### Step 2: Add Remote Repository
If you haven’t already linked your GitHub repository, add it as a remote:
```bash
git remote add origin https://github.com/your-username/your-repository.git
```
Verify the remote link:
```bash
git remote -v
```
#### Step 3: Add and Commit Changes
Add all files to the staging area:
```bash
git add .
```
Commit the changes:

```bash
git commit -m "Your commit message here"
```

#### Step 4: Push to GitHub
Push your code to the main branch of the remote repository:

```bash
git push -u origin main
```






### 2.1.4 Work with Branch
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




### 2.1.5 Pull
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


## 2.2 Using the JupyterHub Git Interface

### 2.2.1 Clone a Repository to JupyterHub
1. Open the **Git** tab in JupyterHub (usually in the right-hand sidebar).
2. Click **Clone Repository**.
3. Paste the HTTPS link of the repository and confirm. The repository will be cloned into your current working directory.


### 2.2.2 Create a Repository
1. Create a new repository on GitHub:
   - Go to [GitHub](https://github.com), click the **+** icon in the top-right corner, and select **New Repository**.
   - Enter the repository name, choose visibility (Public/Private), and click **Create Repository**.
2. Initialize a Git repository in JupyterHub:
   - Open the **Git** tab and click **Initialize Repository** to set up Git for your local project.
   - Use **Commit** to stage and commit changes.
   - Use **Push** to upload the code to the remote repository.


### 2.2.3 Push Changes
1. Stage and commit changes:
   - Open the **Git** tab, select the changed files, add a commit message, and click **Commit**.
2. Push changes:
   - Click **Push** to upload your commits to the remote repository.

### 2.2.4 Pull Changes
1. Pull updates:
   - Click **Pull** in the **Git** tab to fetch and merge changes from the remote repository.


### 2.2.5 Work with Branches
#### Create and Switch Branches
1. Create a branch:
   - In the **Git** tab, go to the Branches section and click the **New Branch** button. Enter the new branch name and confirm.
2. Switch branches:
   - Select the branch you want to work on from the branch dropdown menu.

    > **Note**:  If you have uncommitted changes, make sure to either commit or stash them before switching branches to avoid conflicts.
     
#### Merge Branches
> **Note**: The JupyterHub Git interface does not currently support branch merging. Use the **Terminal** method for this.

#### Delete Branches
> **Note**: The JupyterHub Git interface does not currently support branch deletion. Use the **Terminal** method for this.


---

# 03. Github Workflow
This section summarizes a recommended workflow for using GitHub in a team setting. It includes creating a GitHub account, working with repositories, and collaborating effectively using branches and pull requests.

## 1. Create a GitHub Account
- Sign up on [GitHub](https://github.com) and complete the setup process.
- Familiarize yourself with GitHub's interface and basic functions.

## 2. Generate and Save Your Personal Access Token (PAT)
- Generate a **Personal Access Token (PAT)** following the instructions in section `01`.
- Save the token securely; you'll need it for authentication when pushing or pulling code.

## 3. Create a Team Repository on GitHub
- One team member (team leader) creates a repository for the team.
- The team leader invites other members as collaborators.

## 4. Clone the Course Repository to JupyterHub

## 5. Unlink the Course Repository's Remote and Link to the Team's Repository
- After cloning the course repository, unlink the course's remote repository:
  ```bash
  git remote remove origin
  ```
- Link the local repository to your team's remote repository:
```bash
git remote add origin https://github.com/your-team/team-repository.git
```

- Verify the remote link:
```bash
git remote -v
```

## 6. Run and Extend the Project in JupyterHub
Each team member runs the project in JupyterHub, explores the codebase, and extends their work based on the project goals.

## 7. Create a Branch for Your Work 
Once you've completed a part of the project, create a new branch to isolate your changes:
```bash
git checkout -b your-branch-name
```
Push your branch to the team's remote repository for review:

```bash
git push -u origin your-branch-name
```

## 8. Collaborate on the Branch
Team members review and discuss the branch's work.
Make additional changes if necessary, committing and pushing them to the branch.

## 9. Merge Branch to main
Once the team agrees the branch is ready, merge it into the main branch:
Switch to main:
```bash
git checkout main
```
Merge the branch:
```bash
git merge your-branch-name
```
Push the updated main branch to the remote repository:
```bash
git push origin main
```

## 10. Pull Changes from Your Target Branch Before Pushing
Before pushing your changes, always pull updates from the target branch (the branch you want to push to). This ensures your local branch is synchronized with any updates from the remote branch.

### Steps to Pull from the Target Branch:
- Identify the branch you want to push to (e.g., branch-name).
- Pull the latest updates from the remote branch:
    ```bash
    git pull origin branch-name
    ```
### Key Considerations:
- Only pull if the remote branch has updates: If the remote branch has no updates, you can skip the pull step.
- Resolve conflicts if necessary: If there are any conflicts between your local changes and the pulled updates, Git will notify you. You need to:
    - Open the conflicting files and resolve the conflicts.
    - Mark the conflicts as resolved:
    ```bash
    git add <file-name>
    ```
    Complete the merge process:
    ```
    git commit
    ```

  
## 11. Push Your Work
After pulling the latest updates, push your changes or branch to the remote repository as needed:
```bash
git push
```

---

# 04. Resolving Conflicts and Common Git Issues




---

# 05. Other Useful Git Commands
## 5.1 Clone Only a Specific Folder from a GitHub Repository
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













**Reference**:

https://docs.github.com/en/get-started

https://training.github.com/downloads/github-git-cheat-sheet.pdf


