\# QuantForge Coding Rules



\## Primary Objective



Maintain a production-quality trading platform.



Correctness is always more important than speed.



\---



\# Development Workflow



Every task must follow this sequence:



1\. Read AGENTS.md

2\. Read PROJECT\_STATE.md

3\. Read ARCHITECTURE.md

4\. Explain the implementation plan

5\. Implement only the requested feature

6\. Verify the project still builds

7\. Explain every modified file



Never skip these steps.



\---



\# Scope Rules



One task = One feature.



Do not combine unrelated improvements.



Do not perform "while I'm here" refactoring.



\---



\# File Modification Policy



Preferred:



\- 1 file



Maximum:



\- 2 files



More than 2 files requires architectural justification.



\---



\# Refactoring Policy



Never refactor working modules only because they "look cleaner".



Refactoring is allowed only when:



\- fixing a bug

\- reducing duplication

\- enabling a required feature



\---



\# Code Quality



Always write:



\- readable code

\- explicit names

\- small functions

\- reusable components



Avoid:



\- dead code

\- commented-out code

\- magic numbers

\- duplicated logic



\---



\# Architecture Rules



Business logic belongs only to business modules.



Repositories store data.



Executors execute.



Decision Engine decides.



Consumer orchestrates.



Never mix responsibilities.



\---



\# Database Rules



Never remove existing columns.



Never change schema without migration.



Always preserve backward compatibility.



\---



\# Testing



Before finishing:



\- imports succeed

\- syntax is valid

\- runtime consistency is preserved



If something cannot be verified, explicitly state it.



\---



\# Git Policy



Every completed feature:



git add



↓



git commit



↓



git push



One commit should represent one logical feature.



\---



\# Communication



If requirements are unclear:



Stop.



Explain the uncertainty.



Never invent missing requirements.



\---



\# Final Rule



Stability > Performance > Convenience.

