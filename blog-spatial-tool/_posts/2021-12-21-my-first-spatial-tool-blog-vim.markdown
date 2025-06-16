---
layout: post-wide
title:  "My First Blog on Tools is about Vim and Spacevim"
date:   2021-12-21 20:41:32 +0800
category: Tools
author: Hank Li
---

My first blog on tools is about using [vim](https://www.vim.org/){:target="_blank"} and [spacevim](https://spacevim.org/){:target="_blank"}. It might be a bit surprising, as this blog is about geospatial data science rather than coding. Why am I introducing a text editor as the first blog of geospatial tools? First, I like vim. Second, all blogs are written using vim and spacevim, so I'd like to give vim credit.
I have to acknowledge that I’m, by no means, a Vim expert. I just would like to share something that I am learning and that might be useful to others as well.


### What is Vim?
To learn what vim is, pls see [here](https://opensource.com/resources/what-vim){:target="_blank"}. 
Most deem that Vim is just a text editor like Notepad, Sublime Text, and Atom; some may think vim is mostly about customizing keybindings or IDE interface. 

Well, for me, vim is a whole new way of interacting with computers. I used to heavily depend on GUIs, and I even felt a bit overwhelmed whenever I need to install something via command line tools, but now I like command lines. It is very efficient to locate things and do stuff. 
Vim makes my work more efficient, productive, and fun, although the learning curve is steep, but the time spent on it is worthing, as I learned a whole lot of things that I may never know.

Most data scientists use Python or R in the daily work, so the most poputer IDEs might be JupyterLab and RStudio.
For the purpose of programming with R or Python alone, it is fine to use those tools you are familiar with. However, if one will also write scripts such as ruby, go, markdown, or command line shells like bash and zsh, etc., it is worthing spending some time on vim, because instead of installing different IDEs for different languages, it is more convenient to let one do all the jobs.

### Basic Vim Commands
Before we start to learn vim commends, it is important to understand that Vim has modes:

1. Command Mode: You cannot write text in command mode. When you start Vim, you are placed in Command mode. In this mode, you can locate and modify text such as moving across the screen, deleting and coping text.
2. Insert Mode: When you want to write something on a file, you must enter the insert mode by typing "i" or "a" or some other commands.

To learn more commands, see [Basic Vim Commands](https://vim.rtorr.com/){:target="_blank"}. Do not feel overwhelmed, as we do not need to remember all of them to use vim.
First, [install vim](https://www.vim.org/download.php){:target="_blank"} and simply use keys such as h, j, k, l to move your cursor across the screen. Second, type "i", and write some texts. It is fun. If you are a heavy mouse user, it will be very inconvenient at first, but after a couple of months, you will love vim.
In this process, you may go back to the basic vim commands page quite a lot. 

### Vim and Neovim
See [here](https://blog.devgenius.io/vim-vs-neovim-26b856694566){:target="_blank"} to learn why you should use Neovim. 
When I started to learn vim a couple of years ago, I tried Neovim, but it was not easy to setup neovim as a python IDE, as I had to mannually setup everything from code completion to linting. In this process I learned what basic functionalities IDEs require. 
It was a bit overwhelming though, as I did not even notice this when I was using other editors. 

For vim beginners, it is much easier to start with Spacevim that will be introduced later on. Here, I only list a few things that neovim need to setup for your information. 

1. vim plugin-manager. We need a plugin such as vim-plug or Dein to manage all vim plugins.
2. code completion
3. [linting](https://www.freecodecamp.org/news/what-is-linting-and-how-can-it-save-you-time){:target="_blank"}
4. code formatting
5. Language Server and [Language Server Protocol](https://microsoft.github.io/language-server-protocol/){:target="_blank"}

Most of these have been setup in SpaceVim automatically. Just remind one thing: after installing neovim, use :checkhealth to check whether neovim has Python3 support as well as nodejs and ruby support, as a lot of plugins require these support.

For Python3 support, you may install [pyenv](https://github.com/pyenv/pyenv){:target="_blank"} to manage all python versions. Some may prefer Anaconda to manage python versions and libraries. It is okday. Just ensure that checkhealth shows OK for Python3 support. 

### SpaceVim
It is destined that my favorite editor is related to "space":) But here "space" stands for the key space rather than the "spatiotemporal" space.
The [SpaceVim](https://spacevim.org/documentation/){:target="_blank"} project is inspired by the [spacemacs](https://www.spacemacs.org/){:target="_blank"}. BTW, I like its slogon: 
>The best editor is neither Emacs nor Vim, it's Emacs and Vim!

I like spacevim because of its main features or goals.
1. Mnemonic key bindings navigation
1. More IDE-like features in both Vim and Neovim
1. Better programming language support
1. Cross-platform consistency
1. Fast start up and efficient operation

For elementary Vim users like me, it is very simple to config (much easier than neovim) but we can do deep configuration system.
SpaceVim uses layers to collect related packages together to provides features. For example, the lang#python layer provides auto-completion, syntax checking, and REPL support for python files. This approach helps keep configurations well organized.
The best way to learn spacevim is to read this [documentation](https://spacevim.org/documentation/).

For vim beginners, it is better to use vim as general IDE: a general guide for using SpaceVim as an IDE. see <https://spacevim.org/use-vim-as-ide/>. 
