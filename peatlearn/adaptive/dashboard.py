#!/usr/bin/env python3
"""
Dashboard for PeatLearn Adaptive Learning System
Visualizes user learning progress, patterns, and provides insights
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict, Counter

class Dashboard:
    """
    Creates visualizations and insights for user learning progress
    """
    
    def __init__(self):
        self.color_scheme = {
            'struggling': '#FF6B6B',
            'learning': '#4ECDC4', 
            'advanced': '#45B7D1',
            'new': '#96CEB4',
            'primary': '#667eea',
            'secondary': '#764ba2'
        }
    
    def render_dashboard(self, 
                        user_profile: Dict[str, Any],
                        interactions: List[Dict[str, Any]],
                        quiz_results: List[Dict[str, Any]]) -> None:
        """
        Render the complete learning dashboard
        
        Args:
            user_profile: User's learning profile
            interactions: List of user interactions
            quiz_results: List of quiz results
        """
        
        st.header("📊 Your Learning Journey")
        
        # Overview metrics
        self._render_overview_metrics(user_profile, interactions, quiz_results)
        
        # Profile summary
        self._render_profile_summary(user_profile)
        
        # Learning progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_topic_mastery_chart(user_profile)
            self._render_learning_velocity_chart(interactions)
        
        with col2:
            self._render_interaction_timeline(interactions)
            self._render_quiz_performance_chart(quiz_results)
        
        # Detailed analytics
        with st.expander("📈 Detailed Analytics", expanded=False):
            self._render_detailed_analytics(user_profile, interactions, quiz_results)
        
        # Recommendations
        self._render_recommendations(user_profile)
    
    def _render_overview_metrics(self, 
                                user_profile: Dict[str, Any],
                                interactions: List[Dict[str, Any]],
                                quiz_results: List[Dict[str, Any]]) -> None:
        """Render key performance metrics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_interactions = len(interactions)
            st.metric(
                "Total Interactions",
                total_interactions,
                delta=f"+{min(total_interactions, 5)}" if total_interactions > 0 else None
            )
        
        with col2:
            topics_explored = len(user_profile.get('topic_mastery', {}))
            st.metric(
                "Topics Explored",
                topics_explored,
                delta=f"+{min(topics_explored, 3)}" if topics_explored > 0 else None
            )
        
        with col3:
            if quiz_results:
                avg_quiz_score = sum(r.get('score_percentage', 0) for r in quiz_results) / len(quiz_results)
                st.metric(
                    "Avg Quiz Score",
                    f"{avg_quiz_score:.1f}%",
                    delta=f"+{min(avg_quiz_score - 60, 20):.1f}%" if avg_quiz_score > 60 else None
                )
            else:
                st.metric("Avg Quiz Score", "No quizzes yet", delta=None)
        
        with col4:
            learning_velocity = self._calculate_learning_velocity(interactions)
            st.metric(
                "Learning Velocity",
                f"{learning_velocity:.1f}/day",
                delta=f"+{max(learning_velocity - 1, 0):.1f}" if learning_velocity > 1 else None
            )
    
    def _render_profile_summary(self, user_profile: Dict[str, Any]) -> None:
        """Render user profile summary"""
        
        st.subheader("🎯 Your Learning Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_state = user_profile.get('overall_state', 'new')
            state_color = self.color_scheme.get(overall_state, '#666')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {state_color}22, {state_color}11); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid {state_color};">
                <h4 style="margin: 0; color: {state_color};">Learning State</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold; text-transform: capitalize;">
                    {overall_state}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            learning_style = user_profile.get('learning_style', 'explorer')
            style_icons = {'explorer': '🗺️', 'deep_diver': '🔬', 'balanced': '⚖️'}
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4ECDC422, #4ECDC411); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #4ECDC4;">
                <h4 style="margin: 0; color: #4ECDC4;">Learning Style</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold; text-transform: capitalize;">
                    {style_icons.get(learning_style, '🎯')} {learning_style.replace('_', ' ')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            session_count = user_profile.get('session_patterns', {}).get('total_sessions', 0)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #45B7D122, #45B7D111); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #45B7D1;">
                <h4 style="margin: 0; color: #45B7D1;">Sessions</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {session_count} total
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_topic_mastery_chart(self, user_profile: Dict[str, Any]) -> None:
        """Render topic mastery radar/bar chart"""
        
        st.subheader("📚 Topic Mastery")
        
        topic_mastery = user_profile.get('topic_mastery', {})
        
        if not topic_mastery:
            st.info("Start exploring topics to see your mastery levels!")
            return
        
        # Prepare data
        topics = list(topic_mastery.keys())
        mastery_levels = [topic_mastery[topic]['mastery_level'] * 100 for topic in topics]
        states = [topic_mastery[topic]['state'] for topic in topics]
        
        # Create color mapping
        colors = [self.color_scheme.get(state, '#666') for state in states]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[topic.title() for topic in topics],
                y=mastery_levels,
                marker_color=colors,
                text=[f"{level:.1f}%" for level in mastery_levels],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Mastery: %{y:.1f}%<br>State: %{customdata}<extra></extra>',
                customdata=states
            )
        ])
        
        fig.update_layout(
            title="Topic Mastery Levels",
            xaxis_title="Topics",
            yaxis_title="Mastery Level (%)",
            yaxis=dict(range=[0, 100]),
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_learning_velocity_chart(self, interactions: List[Dict[str, Any]]) -> None:
        """Render learning velocity over time"""
        
        st.subheader("🚀 Learning Velocity")
        
        if not interactions:
            st.info("Start learning to see your velocity!")
            return
        
        # Group interactions by date
        daily_counts = defaultdict(int)
        
        for interaction in interactions:
            try:
                timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                date = timestamp.date()
                daily_counts[date] += 1
            except (ValueError, KeyError):
                continue
        
        if not daily_counts:
            st.info("No timestamp data available")
            return
        
        # Create time series
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        
        # Calculate rolling average
        window_size = min(7, len(dates))
        if len(dates) >= window_size:
            rolling_avg = []
            for i in range(len(dates)):
                start_idx = max(0, i - window_size + 1)
                avg = sum(counts[start_idx:i+1]) / min(window_size, i + 1)
                rolling_avg.append(avg)
        else:
            rolling_avg = counts
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=counts,
            mode='lines+markers',
            name='Daily Interactions',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ))
        
        if len(dates) > 1:
            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_avg,
                mode='lines',
                name=f'{window_size}-day Average',
                line=dict(color='#764ba2', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title="Learning Activity Over Time",
            xaxis_title="Date",
            yaxis_title="Interactions per Day",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_interaction_timeline(self, interactions: List[Dict[str, Any]]) -> None:
        """Render interaction timeline"""
        
        st.subheader("⏰ Recent Activity")
        
        if not interactions:
            st.info("No interactions yet!")
            return
        
        # Show recent interactions
        recent_interactions = sorted(
            interactions, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )[:10]
        
        for i, interaction in enumerate(recent_interactions):
            timestamp = interaction.get('timestamp', 'Unknown time')
            query = interaction.get('user_query', 'Unknown query')
            topic = interaction.get('topic', 'General')
            feedback = interaction.get('user_feedback')
            
            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%m/%d %H:%M")
            except:
                time_str = timestamp[:16] if len(timestamp) > 16 else timestamp
            
            # Feedback emoji
            feedback_emoji = ""
            if feedback == 1:
                feedback_emoji = "👍"
            elif feedback == -1:
                feedback_emoji = "👎"
            
            # Display interaction
            with st.container():
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; border-left: 3px solid #667eea;">
                    <small style="color: #666;">{time_str} • {topic.title()}</small><br>
                    <span style="font-size: 0.9rem;">{query[:80]}{'...' if len(query) > 80 else ''}</span>
                    <span style="float: right;">{feedback_emoji}</span>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_quiz_performance_chart(self, quiz_results: List[Dict[str, Any]]) -> None:
        """Render quiz performance chart"""
        
        st.subheader("🎯 Quiz Performance")
        
        if not quiz_results:
            st.info("Take a quiz to see your performance!")
            return
        
        # Prepare data
        scores = [r.get('score_percentage', 0) for r in quiz_results]
        topics = [r.get('topic', 'General') for r in quiz_results]
        
        # Create performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(scores) + 1)),
            y=scores,
            mode='lines+markers',
            name='Quiz Scores',
            line=dict(color='#45B7D1', width=3),
            marker=dict(size=8, color='#45B7D1'),
            hovertemplate='<b>Quiz %{x}</b><br>Score: %{y:.1f}%<br>Topic: %{customdata}<extra></extra>',
            customdata=topics
        ))
        
        # Add average line
        avg_score = sum(scores) / len(scores)
        fig.add_hline(
            y=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Average: {avg_score:.1f}%"
        )
        
        fig.update_layout(
            title="Quiz Score Progression",
            xaxis_title="Quiz Number",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_analytics(self, 
                                  user_profile: Dict[str, Any],
                                  interactions: List[Dict[str, Any]],
                                  quiz_results: List[Dict[str, Any]]) -> None:
        """Render detailed analytics section"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Interaction Analysis")
            
            # Interaction type distribution
            if interactions:
                interaction_types = [i.get('interaction_type', 'chat') for i in interactions]
                type_counts = Counter(interaction_types)
                
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Interaction Types"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feedback distribution
            if interactions:
                feedback_data = [i.get('user_feedback') for i in interactions if i.get('user_feedback') is not None]
                if feedback_data:
                    positive = sum(1 for f in feedback_data if f > 0)
                    negative = sum(1 for f in feedback_data if f < 0)
                    
                    fig = go.Figure(data=[
                        go.Bar(x=['Positive', 'Negative'], y=[positive, negative],
                              marker_color=['#4ECDC4', '#FF6B6B'])
                    ])
                    fig.update_layout(title="Feedback Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎓 Learning Patterns")
            
            # Session length analysis
            session_patterns = user_profile.get('session_patterns', {})
            avg_session_length = session_patterns.get('avg_session_length', 0)
            
            st.metric("Average Session Length", f"{avg_session_length:.1f} interactions")
            
            # Topic exploration breadth
            topic_mastery = user_profile.get('topic_mastery', {})
            if topic_mastery:
                st.write("**Topic Exploration:**")
                for topic, data in topic_mastery.items():
                    state = data['state']
                    interactions_count = data['total_interactions']
                    st.write(f"• {topic.title()}: {interactions_count} interactions ({state})")
    
    def _render_recommendations(self, user_profile: Dict[str, Any]) -> None:
        """Render personalized recommendations"""
        
        st.subheader("💡 Personalized Recommendations")
        
        from .profile_analyzer import LearnerProfiler
        
        profiler = LearnerProfiler()
        recommendations = profiler.get_recommendations(user_profile)
        
        if not recommendations:
            st.info("Keep learning to get personalized recommendations!")
            return
        
        for rec in recommendations[:3]:  # Show top 3
            priority_color = {
                'high': '#FF6B6B',
                'medium': '#4ECDC4', 
                'low': '#45B7D1'
            }.get(rec.get('priority', 'medium'), '#666')
            
            with st.container():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {priority_color}22, {priority_color}11); 
                            padding: 1rem; margin: 0.5rem 0; border-radius: 10px; border-left: 4px solid {priority_color};">
                    <h4 style="margin: 0; color: {priority_color};">{rec['title']}</h4>
                    <p style="margin: 0.5rem 0 0 0;">{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _calculate_learning_velocity(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate learning velocity (interactions per day)"""
        
        if not interactions:
            return 0.0
        
        # Get date range
        timestamps = []
        for interaction in interactions:
            try:
                timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                timestamps.append(timestamp)
            except (ValueError, KeyError):
                continue
        
        if len(timestamps) < 2:
            return len(interactions)  # All interactions in one day
        
        timestamps.sort()
        days = (timestamps[-1] - timestamps[0]).days + 1
        
        return len(interactions) / days

# Global instance for easy access
dashboard = Dashboard()
