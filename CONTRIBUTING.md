# Contributing

- [Contributing](#contributing)
  - [🛠️ How to contribute](#️-how-to-contribute)
  - [🧭 Where to start?](#-where-to-start)
  - [📢 Guidelines](#-guidelines)
    - [Cards' format](#cards-format)
    - [Images](#images)
  - [⚠️ Significant changes](#️-significant-changes)

## 🛠️ How to contribute

<details>
<summary><b> Quick: on GitHub </b></summary>

1. Select the `src` directory:
   - ![image](https://user-images.githubusercontent.com/88633026/210842064-d5ea1e87-fa4d-497b-baa4-1ee8979cdb6e.png)
2. Select the chapter's file you want to modify:
   - ![image](https://user-images.githubusercontent.com/88633026/210842429-64ae41f6-83de-4479-abd0-b50d8fee3e5f.png)
3. Hit the edit button:
   - ![image](https://user-images.githubusercontent.com/88633026/210845278-910e98ac-0df3-4b5d-a177-15a0c37feb1f.png)
4. Edit the file:
   - ![image](https://user-images.githubusercontent.com/88633026/210845525-3f378dce-a229-4c00-9dde-e5e80fe3bf06.png)
5. Name your changes and propose them:
   - ![image](https://user-images.githubusercontent.com/88633026/210848012-1c6a4bdf-e8ae-45fa-bc98-0fc56a8c30e2.png)
6. Create your pull request:
   1. ![image](https://user-images.githubusercontent.com/88633026/210846256-199e41b0-2712-4004-b8c7-4ab794ef676b.png)
   2. ![image](https://user-images.githubusercontent.com/88633026/210847584-e42c8d24-ec5f-4cc5-afc3-c1e3dbfcdd76.png)

</details>

<details>
<summary><b> Average: locally </b></summary>

Please follow [these steps](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

</details>

<details>
<summary><b> Long: generate flashcards </b></summary>

Requirements:

1. [VSCode](https://code.visualstudio.com/Download) (or [VSCodium](https://vscodium.com/)) >= 1.47
2. [Anki](https://apps.ankiweb.net/) >= 2.1.21
3. [AnkiConnect](https://ankiweb.net/shared/info/2055492159) >= 2020-07-13

Create flashcards:

1. Do what is asked in the [average's section](CONTRIBUTING.md#average-locally)
2. Launch Anki
3. Launch VSCode
4. Download the **Anki for VSCode** extension
5. Do `Ctrl + shift + p` in VSCode
6. Type *Anki*
7. Select **Anki: Sync Anki**.
8. Go on the Anki's window. You should see a pop-up asking you to connect.
9. Create your Anki account.
10. Download these extensions:
    - [Docs Markdown](https://marketplace.visualstudio.com/items?itemName=docsmsft.docs-markdown)
    - [Markdown All in One](https://open-vsx.org/extension/yzhang/markdown-all-in-one)
    - [Paste Image](https://open-vsx.org/extension/mushan/vscode-paste-image)

</details>
<br>

## 🧭 Where to start?

Look at the [progress.md](docs/progress.md).

## 📢 Guidelines

### Cards' format

- Question: a short `## Your question?`
- Answer:
    1. Start with a quick answer (important parts in **bold**)
    2. Develop your answer
    3. At the bottom of your answer add your source
    4. Add an example (facultative)

For example,

```md
## What is a model's architecture?

The architecture is the **functional form of the model**. Indeed, a model can be split into an architecture and parameter(s). The parameters are some variables that define how the architecture operates.

For example, $y=ax+b$ is an architecture with the parameters $a$ and $b$ that change the behavior of the function.

[Source](https://nathanieldamours.github.io/blog/deep%20learning%20for%20coders/jupyter/2021/12/17/dl_for_coders_01.html#Architecture-and-Parameters)
```

### Images

Do not add images unless it is **really** necessary because we have to keep anki decks' size small.

## ⚠️ Significant changes

If you intend to make significant changes, please [open an issue](https://docs.github.com/en/enterprise-cloud@latest/issues/tracking-your-work-with-issues/creating-an-issue) so that we can discuss it first. Remember to [link it to your pull request](https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword).
