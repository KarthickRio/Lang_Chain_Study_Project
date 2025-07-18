"""
Transcript data for the complex extraction agent
Contains sample meeting transcript for testing the extraction workflow
"""

def get_complex_transcript():
    """Return a complex meeting transcript for testing"""
    return """
PROJECT KICKOFF MEETING - Q1 2024 MOBILE APP REDESIGN
Date: January 15, 2024
Duration: 2 hours 30 minutes

[PHASE 1: INTRODUCTIONS & AGENDA REVIEW - 30 minutes]

Sarah_Chen (Project Manager): Good morning everyone. Welcome to the Q1 2024 mobile app redesign project kickoff. I'm Sarah Chen, your project manager. Let's start with quick introductions.

Marcus_Rodriguez (Lead Developer): Marcus Rodriguez, Lead Developer from the mobile team. I'll be overseeing the technical implementation.

Dr_Kim_Patel (UX Research Director): Dr. Kim Patel, UX Research Director. I'll be leading user research and experience design.

Jennifer_Wu (Product Owner): Jennifer Wu, Product Owner. I'll be defining requirements and prioritizing features.

David_Thompson (QA Manager): David Thompson, QA Manager. I'll handle testing strategy and quality assurance.

Lisa_Chang (Marketing Director): Lisa Chang, Marketing Director. I'll coordinate launch activities and user communication.

Alex_Kumar (DevOps Engineer): Alex Kumar, DevOps Engineer. I'll manage deployment and infrastructure.

[PHASE 2: PROJECT OVERVIEW & REQUIREMENTS - 45 minutes]

Sarah_Chen: Our goal is to redesign the mobile app to improve user engagement by 40% and reduce bounce rate by 25%. We have a hard deadline of March 31st for beta release.

Jennifer_Wu: Based on user feedback, our top priorities are: simplified navigation, faster load times, and better accessibility features. We need to support both iOS and Android platforms.

Dr_Kim_Patel: Our research shows users are frustrated with the current search functionality. 67% of users abandon tasks due to poor search results. This is our highest priority fix.

Marcus_Rodriguez: From a technical standpoint, we need to migrate from React Native 0.68 to 0.73, which will require significant refactoring. This impacts our timeline.

David_Thompson: I'm concerned about the March 31st deadline. With the React Native migration, we'll need extra time for regression testing. I recommend pushing beta to April 15th.

Sarah_Chen: That's a valid concern. Let's discuss this further. The marketing campaign is already planned for April 1st launch.

Lisa_Chang: We could do a soft launch on March 31st to selected users, then full release on April 15th. This gives us buffer time for fixes.

**DECISION 1**: Agreed to implement phased launch approach - soft launch March 31st, full release April 15th.

[PHASE 3: TECHNICAL ARCHITECTURE DISCUSSION - 35 minutes]

Marcus_Rodriguez: For the new architecture, I propose using Redux Toolkit for state management and React Query for data fetching. This will improve performance significantly.

Alex_Kumar: We'll need to update our CI/CD pipeline to handle the new build process. I'll need two weeks to set up the new deployment infrastructure.

Dr_Kim_Patel: The new design system requires 15 custom components. We need to ensure they're accessible and meet WCAG 2.1 AA standards.

David_Thompson: Each new component needs comprehensive testing. I suggest we implement automated accessibility testing as part of our CI pipeline.

**DECISION 2**: Adopt Redux Toolkit and React Query for better performance and maintainability.

**ACTION ITEM 1**: Alex to set up new CI/CD pipeline by January 29th (HIGH PRIORITY)
**ACTION ITEM 2**: Dr. Kim to create accessibility testing checklist by January 22nd (MEDIUM PRIORITY)

[PHASE 4: CONFLICT RESOLUTION - 20 minutes]

Jennifer_Wu: I'm concerned about the scope creep. The stakeholders are requesting push notifications, which wasn't in the original spec.

Marcus_Rodriguez: Push notifications will add at least 2 weeks to development time. We'd need to integrate with Firebase and implement proper permission handling.

Sarah_Chen: This is exactly the kind of scope change we need to manage carefully. Jennifer, what's the business justification?

Jennifer_Wu: Marketing claims it could increase user retention by 15%. But I agree it's not essential for MVP.

Lisa_Chang: Actually, we can achieve similar results with email campaigns initially. Push notifications can be phase 2.

**CONFLICT**: Disagreement over including push notifications in MVP scope.
**RESOLUTION**: Push notifications moved to phase 2 post-launch. Focus on core redesign features for MVP.

[PHASE 5: RISK ASSESSMENT & MITIGATION - 25 minutes]

David_Thompson: I see several risks: tight timeline, React Native migration complexity, and team capacity with John out on medical leave.

Marcus_Rodriguez: We're down one senior developer. I suggest bringing in a contractor for the migration work.

Sarah_Chen: Budget allows for contractor, but we need to find someone quickly. Marcus, can you lead the hiring process?

Dr_Kim_Patel: Another risk is user acceptance. The new design is significantly different. We should plan for user training materials.

Lisa_Chang: I'll coordinate with customer success to create onboarding tutorials and help documentation.

**ACTION ITEM 3**: Marcus to hire React Native contractor by January 20th (HIGH PRIORITY)
**ACTION ITEM 4**: Lisa to create user onboarding plan by February 5th (MEDIUM PRIORITY)

[PHASE 6: SPRINT PLANNING & DELIVERABLES - 30 minutes]

Sarah_Chen: Let's break this into 2-week sprints. Sprint 1 focuses on architecture setup and basic navigation.

Jennifer_Wu: Sprint 1 deliverables: Updated navigation structure, basic component library, and search functionality prototype.

Marcus_Rodriguez: Sprint 2 will cover user authentication, profile management, and core feature implementation.

David_Thompson: I need at least 3 days at the end of each sprint for testing. Please factor that into your estimates.

**DECISION 3**: Implement 2-week sprints with 3-day testing buffer at the end of each sprint.

**ACTION ITEM 5**: Sarah to set up sprint tracking in Jira by January 17th (HIGH PRIORITY)
**ACTION ITEM 6**: Jennifer to finalize user stories by January 19th (HIGH PRIORITY)

[PHASE 7: COMMUNICATION & REPORTING - 15 minutes]

Sarah_Chen: We'll have daily standups at 9 AM, sprint reviews every other Friday, and monthly stakeholder updates.

Dr_Kim_Patel: I'll provide weekly UX research updates and user testing results.

Lisa_Chang: Marketing needs weekly progress reports for executive team. I'll coordinate with Sarah on this.

**ACTION ITEM 7**: Sarah to establish weekly reporting schedule by January 16th (MEDIUM PRIORITY)

[MEETING CONCLUSION - 10 minutes]

Sarah_Chen: Great collaboration everyone. Our next meeting is the sprint planning session on January 22nd.

Dr_Kim_Patel: I'm excited about this project. The research data shows real user pain points we can solve.

Marcus_Rodriguez: The technical challenges are significant, but the team is capable. Let's deliver something great.

Jennifer_Wu: Thanks everyone. I'll send out the finalized requirements document by end of day.

David_Thompson: I'll create the testing matrix and share it with the team by Thursday.

Lisa_Chang: Looking forward to launching this improved experience to our users.

**FINAL DECISION**: Project officially approved and initiated. All team members committed to deliverables and timeline.

Meeting adjourned at 11:30 AM.

FOLLOW-UP MEETINGS SCHEDULED:
- Sprint Planning: January 22nd, 2024
- Architecture Review: January 25th, 2024
- Stakeholder Check-in: February 1st, 2024
""" 