import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.comment import Comment
from app.services.sentiment_analyzer import MultilingualSentimentAnalyzer
from app.services.topic_classifier import TopicClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_comments(
    batch_size: int = 32,
    limit: Optional[int] = None,
    force_reanalyze: bool = False,
    use_gpu: bool = False
):  
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS - Multi-language Support")
    print("="*70)
    
    db: Session = next(get_db())
    
    try:
        if force_reanalyze:
            query = select(Comment)
            print("\nMode: Re-analyzing ALL comments")
        else:
            query = select(Comment).where(Comment.sentiment.is_(None))
            print("\nMode: Analyzing only unanalyzed comments")
        
        if limit:
            query = query.limit(limit)
            print(f"Limit: {limit} comments")
        
        comments = db.execute(query).scalars().all()
        total = len(comments)
        
        if total == 0:
            print("\nNo comments to analyze!")
            return
        
        print(f"Found {total} comments to analyze")
        
        print("\n🔧 Initializing analyzers...")
        print(f"   • GPU: {'Enabled' if use_gpu else 'Disabled (CPU mode)'}")
        print(f"   • Batch size: {batch_size}")
        
        analyzer = MultilingualSentimentAnalyzer(use_gpu=use_gpu)
        topic_classifier = TopicClassifier()
        
        print("\nStarting analysis...\n")
        
        analyzed_count = 0
        language_stats = {}
        sentiment_stats = {'positive': 0, 'neutral': 0, 'negative': 0}
        model_stats = {}
        topic_stats = {}
        
        for i in range(0, total, batch_size):
            batch = comments[i:i + batch_size]
            batch_texts = [c.content for c in batch]
            
            results = analyzer.analyze_batch(batch_texts, batch_size=batch_size)
            
            for comment, result in zip(batch, results):
                comment.cleaned_content = result['cleaned_content']
                comment.language = result['language']
                comment.word_count = result['word_count']
                comment.sentiment = result['sentiment']
                comment.sentiment_score = result['sentiment_score']
                comment.analysis_model = result['analysis_model']
                comment.analyzed_at = datetime.now()
                comment.is_valid = result['word_count'] >= 3  # Minimum 3 words
                
                # Topic classification
                topics = topic_classifier.classify_topics(comment.content, comment.language)
                comment.topics = topics if topics else None
                
                # Aspect-level sentiments
                if topics:
                    aspect_sentiments = topic_classifier.get_aspect_sentiments(
                        comment.content,
                        topics,
                        comment.sentiment,
                        comment.language
                    )
                    comment.aspect_sentiments = aspect_sentiments
                    
                    # Update topic stats
                    for topic in topics:
                        topic_stats[topic] = topic_stats.get(topic, 0) + 1
                
                # Update stats
                language_stats[result['language']] = language_stats.get(result['language'], 0) + 1
                sentiment_stats[result['sentiment']] += 1
                model_stats[result['analysis_model']] = model_stats.get(result['analysis_model'], 0) + 1
                
                analyzed_count += 1
            
            db.commit()
            
            progress = (i + len(batch)) / total * 100
            print(f"   Progress: {i + len(batch)}/{total} ({progress:.1f}%) - "
                  f"Last batch: {len(batch)} comments")
        
        print("\nAnalysis completed!")
        print(f"\n{'='*70}")
        print("ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nTotal analyzed: {analyzed_count} comments")
        
        print("\nLanguage Distribution:")
        for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / analyzed_count * 100
            print(f"   • {lang:8s}: {count:4d} ({percentage:5.1f}%)")
        
        print("\nSentiment Distribution:")
        for sentiment, count in sorted(sentiment_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / analyzed_count * 100
            bar = "█" * int(percentage / 2)
            print(f"   • {sentiment:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        print("\nModel Usage:")
        for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / analyzed_count * 100
            print(f"   • {model:15s}: {count:4d} ({percentage:5.1f}%)")
        
        # Topic classification statistics
        if topic_stats:
            print("\nTopic Classification:")
            total_topics = sum(topic_stats.values())
            for topic, count in sorted(topic_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = count / analyzed_count * 100
                bar = "▓" * int(percentage / 3)
                print(f"   • {topic:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
            print(f"   Total topic mentions: {total_topics}")
        
        # Top positive and negative examples
        print("\nSample Results:")
        
        print("\nTop 3 Positive Comments:")
        positive_samples = db.execute(
            select(Comment)
            .where(Comment.sentiment == 'positive')
            .order_by(Comment.sentiment_score.desc())
            .limit(3)
        ).scalars().all()
        
        for idx, comment in enumerate(positive_samples, 1):
            preview = comment.content[:60] + "..." if len(comment.content) > 60 else comment.content
            print(f"   {idx}. [{comment.language}] {preview}")
            print(f"      Score: {comment.sentiment_score:.3f} | Model: {comment.analysis_model}")
        
        print("\nTop 3 Negative Comments:")
        negative_samples = db.execute(
            select(Comment)
            .where(Comment.sentiment == 'negative')
            .order_by(Comment.sentiment_score.desc())
            .limit(3)
        ).scalars().all()
        
        for idx, comment in enumerate(negative_samples, 1):
            preview = comment.content[:60] + "..." if len(comment.content) > 60 else comment.content
            print(f"   {idx}. [{comment.language}] {preview}")
            print(f"      Score: {comment.sentiment_score:.3f} | Model: {comment.analysis_model}")
        
        print(f"\n{'='*70}")
        print("Sentiment analysis completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        db.rollback()
        raise
    
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser(
        description="Analyze comments with multi-language sentiment analysis"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Number of comments to process in one batch (default: 32)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of comments to analyze (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-analyze already analyzed comments'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (requires CUDA)'
    )
    
    args = parser.parse_args()
    
    print("\nStarting sentiment analysis...")
    print("Configuration:")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Limit: {args.limit if args.limit else 'None (all comments)'}")
    print(f"   • Force re-analyze: {args.force}")
    print(f"   • GPU: {args.gpu}")
    
    analyze_comments(
        batch_size=args.batch_size,
        limit=args.limit,
        force_reanalyze=args.force,
        use_gpu=args.gpu
    )
